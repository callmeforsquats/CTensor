#include "backward.h"
#include "../backend/backend.h"
#include "../core/autograd.h"
#include "../core/tensor.h"
#include "../src/backend/backend.h"
#include "../utils/utils.h"
#include <math.h>
#include <stdlib.h>

void free_ReduceCtx(void *ctx) {
    tensor_free(((ReduceCtx *)ctx)->mask);
    free(ctx);
}

void backward_zeros(OpNode *Node) {}

void backward_ones(OpNode *Node) {
    for (int_t i = 0; i < Node->n_inputs; ++i) {
        check_grad(Node->inputs[i]);
        Tensor t =
            make_broadcast_view(Node->inputs[i]->grad, Node->output->shape, Node->output->ndims);
        GET_BACKEND(t.storage->device)->back_add_kernel(&t, Node->output->grad);
    }
}

void backward_view(OpNode *Node) {
    Tensor *in = Node->inputs[0], *output_grad = Node->output->grad;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        Tensor *out_g = tensor_view(output_grad, in->shape, in->ndims, false);
        backend->back_add_kernel(in->grad, out_g);
        tensor_free(out_g);
    }
}

void backward_slice(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        SliceCtx *ctx = (SliceCtx *)Node->ctx;
        Tensor *in_t = tensor_slice(in->grad, ctx->slices, ctx->n_slices);
        backend->back_add_kernel(in_t, out->grad);
        tensor_free(in_t);
    }
}

void backward_transpose(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        TransposeCtx *ctx = (TransposeCtx *)Node->ctx;
        Tensor *out_g = tensor_transpose(out, ctx->order, false);
        backend->back_add_kernel(in->grad, out_g);
        tensor_free(out_g);
    }
}

void backward_add(OpNode *Node) {
    Tensor *l = Node->inputs[0], *r = Node->inputs[1], *out = Node->output;
    BACKEND *backend = GET_BACKEND(l->storage->device);
    if (l->requires_grad) {
        check_grad(l);
        Tensor l_g = make_broadcast_view(l->grad, out->shape, out->ndims);
        backend->back_add_kernel(&l_g, out->grad);
    }
    if (r->requires_grad) {
        check_grad(r);
        Tensor r_g = make_broadcast_view(r->grad, out->shape, out->ndims);
        backend->back_add_kernel(&r_g, out->grad);
    }
}

void backward_sub(OpNode *Node) {
    Tensor *l = Node->inputs[0], *r = Node->inputs[1], *out = Node->output;
    BACKEND *backend = GET_BACKEND(l->storage->device);
    if (l->requires_grad) {
        check_grad(l);
        Tensor l_g = make_broadcast_view(l->grad, out->shape, out->ndims);
        backend->back_add_kernel(&l_g, out->grad);
    }
    if (r->requires_grad) {
        check_grad(r);
        Tensor r_g = make_broadcast_view(r->grad, out->shape, out->ndims);
        backend->back_neg_kernel(&r_g, out->grad);
    }
}
void backward_mul(OpNode *Node) {
    Tensor *l = Node->inputs[0], *r = Node->inputs[1], *out = Node->output;
    BACKEND *backend = GET_BACKEND(l->storage->device);
    if (l->requires_grad) {
        check_grad(l);
        Tensor r_t = make_broadcast_view(r, out->shape, out->ndims);
        Tensor l_g = make_broadcast_view(l->grad, out->shape, out->ndims);
        backend->back_mul_kernel(&l_g, out->grad, &r_t);
    }
    if (r->requires_grad) {
        check_grad(r);
        Tensor l_t = make_broadcast_view(l, out->shape, out->ndims);
        Tensor r_g = make_broadcast_view(r->grad, out->shape, out->ndims);
        backend->back_mul_kernel(&r_g, out->grad, &l_t);
    }
}
void backward_div(OpNode *Node) {
    Tensor *l = Node->inputs[0], *r = Node->inputs[1], *out = Node->output;
    BACKEND *backend = GET_BACKEND(l->storage->device);
    if (l->requires_grad) {
        check_grad(l);
        Tensor r_t = make_broadcast_view(r, out->shape, out->ndims);
        Tensor l_g = make_broadcast_view(l->grad, out->shape, out->ndims);
        backend->back_l_div_kernel(&l_g, out->grad, &r_t);
    }
    if (r->requires_grad) {
        check_grad(r);
        Tensor r_t = make_broadcast_view(r, out->shape, out->ndims);
        Tensor r_g = make_broadcast_view(r->grad, out->shape, out->ndims);
        backend->back_r_div_kernel(&r_g, out->grad, &r_t, out);
    }
}

void backward_neg(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        backend->back_neg_kernel(in->grad, out->grad);
    }
}

void backward_sin(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        backend->back_sin_kernel(in->grad, out->grad, in);
    }
}

void backward_cos(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        backend->back_cos_kernel(in->grad, out->grad, in);
    }
}

void backward_tg(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        backend->back_tg_kernel(in->grad, out->grad, out);
    }
}
void backward_tgh(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        backend->back_tgh_kernel(in->grad, out->grad, out);
    }
}

void backward_exp(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        backend->back_exp_kernel(in->grad, out->grad, out);
    }
}
void backward_ln(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        backend->back_ln_kernel(in->grad, out->grad, in);
    }
}
void backward_relu(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        backend->back_relu_kernel(in->grad, out->grad, out);
    }
}
void backward_sqr(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        backend->back_sqr_kernel(in->grad, out->grad, in);
    }
}

void backward_inv(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        backend->back_inv_kernel(in->grad, out->grad, out);
    }
}
void backward_sqrt(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        backend->back_sqrt_kernel(in->grad, out->grad, out);
    }
}

void backward_log_const(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        float c = ((ConstCtx *)Node->ctx)->C;
        backend->back_c_log_kernel(in->grad, out->grad, in, c);
    }
}

void backward_mul_const(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        float c = ((ConstCtx *)Node->ctx)->C;
        backend->back_c_mul_kernel(in->grad, out->grad, c);
    }
}
void backward_div_const(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        float c = ((ConstCtx *)Node->ctx)->C;
        backend->back_c_mul_kernel(in->grad, out->grad, 1.f / c);
    }
}
void backward_pow_const(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        float c = ((ConstCtx *)Node->ctx)->C;
        if (c == 0) backend->back_add_kernel(in->grad, out->grad);
        else backend->back_c_pow_kernel(in->grad, out->grad, in, c);
    }
}

void backward_sum(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        ReduceCtx *ctx = (ReduceCtx *)Node->ctx;
        Tensor out_g;
        calc_reduce_shape_unsq(in, out_g.shape, ctx->axes, ctx->n_axes);
        calc_strides(out_g.shape, out_g.strides, in->ndims);
        set_tensor_properties(&out_g, NULL, NULL, in->ndims, out->grad->numel, out->grad->offset,
                              out->grad->storage);
        out_g = make_broadcast_view(&out_g, in->shape, in->ndims);
        backend->sum_kernel(&out_g, in->grad);
    }
}

void backward_minmax(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        ReduceCtx *ctx = (ReduceCtx *)Node->ctx;
        Tensor *mask = ctx->mask, out_g;
        // TODO:
        tensor_print(mask);
        tensor_print(out->grad);
        calc_reduce_shape_unsq(in, out_g.shape, ctx->axes, ctx->n_axes);
        calc_strides(out_g.shape, out_g.strides, in->ndims);
        set_tensor_properties(&out_g, NULL, NULL, in->ndims, out->grad->numel, out->grad->offset,
                              out->grad->storage);
        out_g = make_broadcast_view(&out_g, in->shape, in->ndims);
        backend->back_mul_kernel(in->grad, &out_g, mask);
        tensor_print(in->grad);
    }
}

void backward_mean(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        ReduceCtx *ctx = (ReduceCtx *)Node->ctx;
        Tensor out_g;
        int_t N = calc_numel_on_axes(in->shape, in->ndims, ctx->axes, ctx->n_axes);
        calc_reduce_shape_unsq(in, out_g.shape, ctx->axes, ctx->n_axes);
        calc_strides(out_g.shape, out_g.strides, in->ndims);
        set_tensor_properties(&out_g, NULL, NULL, in->ndims, out->grad->numel, out->grad->offset,
                              out->grad->storage);
        out_g = make_broadcast_view(&out_g, in->shape, in->ndims);
        backend->back_c_mul_kernel(in->grad, &out_g, 1.f / N);
    }
}

void backward_dropout(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        DropoutCtx *ctx = (DropoutCtx *)Node->ctx;
        backend->set_seed(ctx->seed);
        backend->back_dropout(in->grad, out->grad, ctx->p);
    }
}
void backward_softmax(OpNode *Node) {
    Tensor *in = Node->inputs[0], *out = Node->output;
    BACKEND *backend = GET_BACKEND(in->storage->device);
    if (in->requires_grad) {
        check_grad(in);
        ReduceCtx *ctx = (ReduceCtx *)Node->ctx;
        int_t shape[MAX_DIM];
        calc_reduce_shape_unsq(in, shape, ctx->axes, ctx->n_axes);
        Tensor *sum = tensor_create_empty(shape, in->ndims, in->storage->device);
        backend->fill_value_kernel(sum, 0.f);
        Tensor sum_t = make_broadcast_view(sum, in->shape, in->ndims);
        backend->back_softmax(in->grad, out->grad, out, &sum_t);
        tensor_free(sum);
    }
}

void backward_matmul(OpNode *Node) {
    Tensor *a = Node->inputs[0], *b = Node->inputs[1], *output_grad = Node->output->grad;
    DEVICE d = a->storage->device;
    BACKEND *backend = GET_BACKEND(d);
    if (a->requires_grad) {
        check_grad(a);
        int_t shape[MAX_DIM], ndims;
        calc_matmul_result_shape(a, b, shape, &ndims);
        Tensor *out_grad = tensor_view(output_grad, shape, ndims, false);
        Tensor b_view = make_matmul_broadcast_view(b, shape, ndims, false);
        tensor_transpose_last2(&b_view, true);
        calc_matmul_result_shape(out_grad, &b_view, shape, &ndims);
        Tensor *tmp = tensor_create_empty(shape, ndims, d);
        backend->fill_value_kernel(tmp, 0.f);
        backend->matmul_kernel(out_grad, &b_view, tmp);
        matmul_squeeze_shape(a, NULL, tmp->shape, tmp->strides, &tmp->ndims);
        backend->sum_kernel(tmp, a->grad);
        tensor_free(out_grad), tensor_free(tmp);
    }
    if (b->requires_grad) {
        check_grad(b);
        int_t shape[MAX_DIM], ndims;
        calc_matmul_result_shape(a, b, shape, &ndims);
        Tensor *out_grad = tensor_view(output_grad, shape, ndims, false);
        Tensor a_view = make_matmul_broadcast_view(a, shape, ndims, true);
        tensor_transpose_last2(&a_view, true);
        calc_matmul_result_shape(&a_view, out_grad, shape, &ndims);
        Tensor *tmp = tensor_create_empty(shape, ndims, d);
        backend->fill_value_kernel(tmp, 0.f);
        backend->matmul_kernel(&a_view, out_grad, tmp);
        matmul_squeeze_shape(NULL, b, tmp->shape, tmp->strides, &tmp->ndims);
        backend->sum_kernel(tmp, b->grad);
        tensor_free(out_grad), tensor_free(tmp);
    }
}