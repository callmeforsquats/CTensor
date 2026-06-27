#include "autograd.h"
#include "../backend/backend.h"
#include "../utils/utils.h"
#include "stdlib.h"
#include "tensor.h"
#include <stdio.h>

#define MAX_SIZE 1024

void op_free(OpNode *Node) {
    for (int_t i = 0; i < Node->n_inputs; ++i)
        tensor_free(Node->inputs[i]);
    free(Node->inputs);
    if (Node->ctx && Node->free_ctx) Node->free_ctx(Node->ctx);
    free(Node);
}

static void toposort(Tensor *root, Tensor **LIST, int_t *count) {
    if (!root || root->grad_visited || !root->creator) return;

    typedef struct {
        Tensor *t;
        int_t input_idx;
    } Node;

    Node *stack = (Node *)malloc(MAX_SIZE * sizeof(Node));
    int_t stack_idx = 0;
    stack[stack_idx] = (Node){root, 0};
    root->grad_visited = true;
    while (stack_idx >= 0) {
        Node *cur = &stack[stack_idx];
        OpNode *op = cur->t->creator;
        if (op && cur->input_idx < op->n_inputs) {
            Tensor *input = op->inputs[cur->input_idx++];
            if (input->requires_grad && input->creator && !input->grad_visited) {
                input->grad_visited = true;
                stack[++stack_idx] = (Node){input, 0};
            }
        } else {
            LIST[(*count)++] = cur->t;
            stack_idx--;
        }
    }
    free(stack);
}

void op_create(Tensor *output, Tensor **inputs, int_t n_inputs, back_fun *grad_fn, char *name,
               void *ctx, void (*free_ctx)(void *)) {

    OpNode *Node = (OpNode *)calloc(1, sizeof(OpNode));
    output->creator = Node;
    Node->output = output;
    Node->backward = grad_fn;
    Node->n_inputs = n_inputs;
    Node->name = name;
    Node->ctx = ctx;
    Node->free_ctx = free_ctx;
    Node->inputs = (Tensor **)malloc(n_inputs * sizeof(Tensor *));
    for (int_t i = 0; i < n_inputs; ++i) {
        Node->inputs[i] = inputs[i];
        inputs[i]->ref_count++;
    }
}

static void print_op_recursive(OpNode *Node, int_t *count, int_t level) {
    if (!Node) return;
    (*count)++;
    printf("===============================================================================\n");
    printf("[%d] OP: %-10s \n| Output: ", *count, Node->name);
    print_tensor_info(Node->output);
    printf("\n");
    for (int_t i = 0; i < Node->n_inputs; ++i) {
        printf("| Input %d: ", i);
        print_tensor_info(Node->inputs[i]);
    };
    printf("===================================================================================\n");
    for (int_t i = 0; i < Node->n_inputs; ++i)
        print_op_recursive(Node->inputs[i]->creator, count, level + 1);
}

void grad_print(Tensor *loss) {
    printf("=========================== GRAPH ==============================================\n");
    int_t count = 0, level = 0;
    print_op_recursive(loss->creator, &count, level);
}
void backward(Tensor *loss) {
    BACKEND *backend = GET_BACKEND(loss->storage->device);
    loss->grad = tensor_create_empty(loss->shape, loss->ndims, loss->storage->device);
    backend->fill_value_kernel(loss->grad, 1.f);
    Tensor **LIST = (Tensor **)malloc(sizeof(Tensor *) * MAX_SIZE);
    int_t count = 0;
    toposort(loss, LIST, &count);
    for (int_t i = count - 1; i >= 0; --i) {
        OpNode *Node = LIST[i]->creator;
        if (Node) Node->backward(Node);
    }
}
void graph_clear(Tensor *loss) { tensor_free(loss); }

void reduce_grad(Tensor *src, Tensor *dst) {
    if (tuples_equal(src->shape, dst->shape, src->ndims, dst->ndims)) {
        GET_BACKEND(src->storage->device)->add_kernel(src, dst, dst);
    } else {
        GET_BACKEND(src->storage->device)->sum_kernel(src, dst);
    }
}

void check_grad(Tensor *input) {
    if (input->grad == NULL) {
        BACKEND *backend = GET_BACKEND(input->storage->device);
        input->grad = tensor_create_empty(input->shape, input->ndims, input->storage->device);
        backend->fill_value_kernel(input->grad, 0.f);
    }
}