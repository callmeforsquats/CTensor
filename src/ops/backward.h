#pragma once
#include "../core/autograd.h"

back_fun backward_view, backward_slice, backward_transpose,

    backward_add, backward_sub, backward_mul, backward_div, backward_zeros, backward_ones,

    backward_neg, backward_sin, backward_cos, backward_tg, backward_tgh, backward_exp, backward_ln,
    backward_sqr, backward_sqrt, backward_relu, backward_inv,

    backward_mul_const, backward_div_const, backward_pow_const, backward_log_const,

    backward_sum, backward_mean, backward_minmax,

    backward_dropout, backward_softmax,

    backward_matmul;

typedef struct {
    int_t order[MAX_DIM];
} TransposeCtx;

typedef struct {
    Slice slices[MAX_DIM];
    int_t n_slices;
} SliceCtx;

typedef struct {
    float C;
} ConstCtx;

typedef struct {
    Tensor *mask;
    int_t axes[MAX_DIM], n_axes;
} ReduceCtx;

typedef struct {
    uint64_t seed;
    float p;
} DropoutCtx;

typedef struct {
    union {
        struct {
            int_t order[MAX_DIM];
        } transpose;
        struct {
            float C;
        } scalar;
        struct {
            Tensor *mask;
            int_t axes[MAX_DIM], n_axes;
        } reduce;
        struct {
            Slice slices[MAX_DIM];
            int_t n_slices;
        } slice;
    };
} OpCTX;

void free_ReduceCtx(void *ctx);