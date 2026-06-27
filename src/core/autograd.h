#pragma once
#include "tensor.h"

typedef void back_fun(OpNode *);

typedef struct OpNode {
    back_fun *backward;
    Tensor **inputs;
    int_t n_inputs;
    Tensor *output;
    char *name;
    void (*free_ctx)(void *);
    void *ctx;
} OpNode;

typedef struct OpTape {
    OpNode *nodes;
    int_t count;
    int_t capacity;
} OpTape;

void backward(Tensor *loss);
void check_grad(Tensor *input);
void reduce_grad(Tensor *src, Tensor *dst);
void op_create(Tensor *output, Tensor **inputs, int_t n_inputs, back_fun *grad_fn, char *name,
               void *ctx, void (*free_ctx)(void *));
void grad_print(Tensor *loss);
void graph_clear(Tensor *loss);