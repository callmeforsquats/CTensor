#include "../core/tensor.h"
#include "../utils/error.h"
#include "../utils/utils.h"
#include "backend.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__clang__)
#define msg "Clang\n"
#elif defined(_MSV_VER)
#define msg ("msvc")
#endif

// ----------------OPERATION CONTEXT----------------

// ------------------------ITERATOR----------------------------
typedef struct {
    int_t coords[MAX_DIM];
    int_t offsets[MAX_OPERANDS];
} TrenIter;

static FORCE_INLINE TrenIter iter_new(const Tensor **items, int_t n_items) {
    TrenIter it;
    for (int_t i = 0; i < MAX_DIM; ++i)
        it.coords[i] = 0;
    for (int_t i = 0; i < n_items; ++i)
        it.offsets[i] = items[i]->offset;
    return it;
}

static FORCE_INLINE void iter_step(TrenIter *it, const int_t *shape, const int_t **strides,
                                   int_t ndims, int_t n_items) {
    for (int_t d = ndims - 1; d >= 0; --d) {
        if (++(it->coords[d]) < shape[d]) {
            for (int_t i = 0; i < n_items; ++i)
                it->offsets[i] += strides[i][d];
            break;
        }
        it->coords[d] = 0;
        for (int_t i = 0; i < n_items; ++i)
            it->offsets[i] -= (shape[d] - 1) * strides[i][d];
    }
}

// ----------------RANDOM OPS--------------------------------

static uint64_t SEED = 1234567890ULL;

static inline void set_seed(uint64_t seed) { SEED = seed; }
static inline uint64_t get_seed() { return SEED; }

static FORCE_INLINE uint64_t xor_gen() {
    uint64_t x = SEED;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return SEED = x;
}

static FORCE_INLINE float next_float() { return (float)xor_gen() / 18446744073709551615.0f; }
static FORCE_INLINE float next_uniform(float min, float max) {
    return min + next_float() * (max - min);
}
static FORCE_INLINE float next_normal(float mean, float std) {
    float sum = 0.f;
    for (int i = 0; i < 12; ++i)
        sum += next_float();
    return (sum - 6.f) * std + mean;
}
//--------------------FILL KERNELS-----------------------------------

static void fill_value_kernel(Tensor *t, const float value) {
    if (value == 0.f) {
        memset(t->storage->data, 0, t->storage->size * sizeof(float));
    } else {
        float *ptr = t->storage->data;
        for (int i = 0; i < t->storage->size; ++i) {
            *(ptr++) = value;
        }
    }
}

static void fill_range_kernel(Tensor *t, int_t start, int_t step) {
    for (int_t i = 0; i < t->numel; ++i) {
        t->storage->data[i] = start;
        start += step;
    }
}

static void fill_from_array_kernel(Tensor *t, const float *arr) {
    memcpy(t->storage->data, arr, t->numel * sizeof(float));
}

static void fill_random_uniform(Tensor *A, float min, float max) {
    float *data = A->storage->data + A->offset;
    for (int_t i = 0; i < A->numel; ++i)
        data[i] = next_uniform(min, max);
}
static void fill_random_normal(Tensor *A, float mean, float std) {
    float *data = A->storage->data + A->offset;
    for (int_t i = 0; i < A->numel; ++i)
        data[i] = next_normal(mean, std);
}

// -------------------------------------UNARY KERNELS---------------------------------------

static FORCE_INLINE void red_multi(float **d) {
    if (*d[1]) *d[1] += *d[0];
    if (*d[2]) *d[2] += *d[0] * *d[0];
    if (*d[3]) *d[3] = *d[0] > *d[3] ? *d[0] : *d[3];
    if (*d[4]) *d[4] = *d[0] < *d[4] ? *d[0] : *d[4];
}
static FORCE_INLINE void sftmx_1(float **d) {
    if (*d[0] > *d[2]) {
        *d[1] = *d[1] * expf(*d[2] - *d[0]) + 1.f;
        *d[2] = *d[0];
    } else {
        *d[1] += expf(*d[0] - *d[2]);
    }
}
static FORCE_INLINE void sum_prod(float **d, OpCtx ctx) {
    float el = 1.f;
    for (int_t i = 0; i < ctx.N; ++i)
        el *= *d[i + 1];
    *d[0] += el;
}

static FORCE_INLINE void apply_op(float **d, OpCtx ctx, OpType op) {
    switch (op) {

    case RAND_UN   : *d[0] = next_uniform(ctx.randu.min, ctx.randu.max); break;
    case RAND_NORM : *d[0] = next_normal(ctx.randn.mean, ctx.randn.std); break;
    case DROPOUT   : *d[1] = next_float() > ctx.dropout.p ? *d[0] * ctx.dropout.inv_p : 0.f; break;
    // ------BINARY---------------
    case BIN_ADD   : *d[2] = *d[0] + *d[1]; break;
    case BIN_SUB   : *d[2] = *d[0] - *d[1]; break;
    case BIN_MUL   : *d[2] = *d[0] * *d[1]; break;
    case BIN_DIV   : *d[2] = *d[0] / *d[1]; break;
    case BIN_LT    : *d[2] = *d[0] < *d[1]; break;
    case BIN_LE    : *d[2] = *d[0] <= *d[1]; break;
    case BIN_EQ    : *d[2] = fabsf(*d[0] - *d[1]) < 1e-9; break;
    case BIN_NE    : *d[2] = *d[0] != *d[1]; break;
    case BIN_GE    : *d[2] = *d[0] >= *d[1]; break;
    case BIN_GT    : *d[2] = *d[0] > *d[1]; break;
    case BIN_ASSIGN: *d[1] = *d[0]; break;
    // ---------UNARY-----------------
    case UN_NEG    : *d[1] = -*d[0]; break;
    case UN_SIN    : *d[1] = sinf(*d[0]); break;
    case UN_COS    : *d[1] = cosf(*d[0]); break;
    case UN_TG     : *d[1] = tanf(*d[0]); break;
    case UN_TGH    : *d[1] = tanhf(*d[0]); break;
    case UN_EXP    : *d[1] = expf(*d[0]); break;
    case UN_LN     : *d[1] = logf(*d[0]); break;
    case UN_SQRT   : *d[1] = sqrtf(*d[0]); break;
    case UN_SQR    : *d[1] = *d[0] * *d[0]; break;
    case UN_INV    : *d[1] = 1.f / *d[0]; break;
    case UN_RELU   : *d[1] = *d[0] > 0 ? *d[0] : 0; break;
    //---------CONST------------------
    case CONST_ADD : *d[1] = *d[0] + ctx.scalar; break;
    case CONST_SUB : *d[1] = *d[0] - ctx.scalar; break;
    case CONST_MUL : *d[1] = *d[0] * ctx.scalar; break;
    case CONST_DIV : *d[1] = *d[0] / ctx.scalar; break;
    case CONST_LT  : *d[1] = *d[0] < ctx.scalar; break;
    case CONST_LE  : *d[1] = *d[0] <= ctx.scalar; break;
    case CONST_EQ  : *d[1] = fabsf(*d[0] - ctx.scalar) < 1e-9; break;
    case CONST_NE  : *d[1] = *d[0] != ctx.scalar; break;
    case CONST_GE  : *d[1] = *d[0] >= ctx.scalar; break;
    case CONST_GT  : *d[1] = *d[0] > ctx.scalar; break;
    case CONST_POW : *d[1] = powf(*d[0], ctx.scalar); break;
    case CONST_LOG : *d[1] = logf(*d[0]) / logf(ctx.scalar);
    // --------------REDUCE--------------
    case RD_PROD   : *d[1] *= *d[0]; break;
    case RD_SUM    : *d[1] += *d[0]; break;
    case RD_MAX    : *d[1] = *d[1] > *d[0] ? *d[1] : *d[0]; break;
    case RD_MIN    : *d[1] = *d[1] < *d[0] ? *d[1] : *d[0]; break;
    // --------------OTHER UTILS---------------------
    case RD_MULTI  : red_multi(d); break;
    case _SFTMX_1  : sftmx_1(d); break;
    case _SFTMX_2  : *d[1] = expf(*d[0] - *d[2]) / *d[3]; break;
    case SUM_PROD  : sum_prod(d, ctx); break;
    case MSE       : *d[2] += (*d[0] - *d[1]) * (*d[0] - *d[1]) * ctx.scalar; break;
    // ------------BACKWARD BINARY----------------------
    case BW_ADD    : *d[0] += *d[1]; break;
    case BW_NEG    : *d[0] -= *d[1]; break;
    case BW_MUL    : *d[0] += *d[1] * *d[2]; break;
    case BW_L_DIV  : *d[0] += *d[1] / *d[2]; break;
    case BW_R_DIV  : *d[0] -= *d[1] * *d[3] / *d[2]; break;
    // ------------BACKWARD UNARY---------------------
    case BW_SIN    : *d[0] += *d[1] * cosf(*d[2]); break;
    case BW_COS    : *d[0] -= *d[1] * sinf(*d[2]); break;
    case BW_TG     : *d[0] += *d[1] * (1.f + *d[2] * *d[2]); break;
    case BW_TGH    : *d[0] += *d[1] * (1.f - *d[2] * *d[2]); break;
    case BW_EXP    : *d[0] += *d[1] * *d[2]; break;
    case BW_LN     : *d[0] += *d[1] / *d[2]; break;
    case BW_SQR    : *d[0] += *d[1] * 2.f * *d[2]; break;
    case BW_SQRT   : *d[0] -= *d[1] / *d[2]; break;
    case BW_INV    : *d[0] -= *d[1] * *d[2] * *d[2]; break;
    case BW_RELU   : *d[0] += *d[2] > 0 ? *d[1] : 0.f; break;
    // --------------BACKWARD CONST----------------------------
    case BW_C_MUL  : *d[0] += *d[1] * ctx.scalar; break;
    case BW_C_POW  : *d[0] += *d[1] * ctx.scalar * powf(*d[2], ctx.scalar - 1.f); break;
    case BW_C_LOG  : *d[0] += *d[1] * ctx.scalar / *d[2]; break;

    case BW_DROPOUT: *d[0] += next_float() > ctx.dropout.p ? *d[1] * ctx.dropout.inv_p : 0.f; break;
    case BW_SOFTMAX: *d[0] = *d[2] * (*d[1] - *d[3]); break;
    case BW_MSE    : *d[0] += *d[1] * 2.f * (*d[2] - *d[3]) * ctx.scalar; break;
    default        : break;
    }
}

static FORCE_INLINE void traversal_kernel(Tensor **items, int_t n_items, OpCtx ctx, OpType op) {
    float *data[MAX_OPERANDS];
    for (int_t i = 0; i < n_items; ++i)
        data[i] = items[i]->storage->data;

    if (all_contiguous(items, n_items)) {
        for (int_t i = 0; i < n_items; ++i)
            data[i] += items[i]->offset;
        for (int_t i = 0; i < items[0]->numel; ++i) {
            float *cur[MAX_OPERANDS];
            for (int_t t = 0; t < n_items; ++t)
                cur[t] = &data[t][i];
            apply_op(cur, ctx, op);
        }
    } else {
        TrenIter it = iter_new(items, n_items);
        int_t *strides[MAX_OPERANDS];
        for (int_t i = 0; i < n_items; ++i)
            strides[i] = items[i]->strides;
        for (int_t i = 0; i < items[0]->numel; ++i) {
            float *cur[MAX_OPERANDS];
            for (int_t t = 0; t < n_items; ++t)
                cur[t] = &data[t][it.offsets[t]];
            apply_op(cur, ctx, op);
            iter_step(&it, items[0]->shape, strides, items[0]->ndims, n_items);
        }
    }
}

// ------------------BINARY KERNELS------------------------

static void add_kernel(Tensor *left, Tensor *right, Tensor *res) {
    traversal_kernel((Tensor *[]){left, right, res}, 3, (OpCtx){0}, BIN_ADD);
}
static void sub_kernel(Tensor *left, Tensor *right, Tensor *res) {
    traversal_kernel((Tensor *[]){left, right, res}, 3, (OpCtx){0}, BIN_SUB);
}
static void mul_kernel(Tensor *left, Tensor *right, Tensor *res) {
    traversal_kernel((Tensor *[]){left, right, res}, 3, (OpCtx){0}, BIN_MUL);
}
static void div_kernel(Tensor *left, Tensor *right, Tensor *res) {
    traversal_kernel((Tensor *[]){left, right, res}, 3, (OpCtx){0}, BIN_DIV);
}
static void lt_kernel(Tensor *left, Tensor *right, Tensor *res) {
    traversal_kernel((Tensor *[]){left, right, res}, 3, (OpCtx){0}, BIN_LT);
}
static void le_kernel(Tensor *left, Tensor *right, Tensor *res) {
    traversal_kernel((Tensor *[]){left, right, res}, 3, (OpCtx){0}, BIN_LE);
}
static void eq_kernel(Tensor *left, Tensor *right, Tensor *res) {
    traversal_kernel((Tensor *[]){left, right, res}, 3, (OpCtx){0}, BIN_EQ);
}
static void ne_kernel(Tensor *left, Tensor *right, Tensor *res) {
    traversal_kernel((Tensor *[]){left, right, res}, 3, (OpCtx){0}, BIN_NE);
}
static void ge_kernel(Tensor *left, Tensor *right, Tensor *res) {
    traversal_kernel((Tensor *[]){left, right, res}, 3, (OpCtx){0}, BIN_GE);
}
static void gt_kernel(Tensor *left, Tensor *right, Tensor *res) {
    traversal_kernel((Tensor *[]){left, right, res}, 3, (OpCtx){0}, BIN_GT);
}

// --------------------UNARY KERNELS---------------------------

static void assign_kernel(Tensor *src, Tensor *dst) {
    if (all_contiguous((Tensor *[]){src, dst}, 2))
        memcpy(dst->storage + dst->offset, src->storage + src->offset, src->numel * sizeof(float));
    else traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){0}, BIN_ASSIGN);
}
static void neg_kernel(Tensor *src, Tensor *dst) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){0}, UN_NEG);
}
static void sin_kernel(Tensor *src, Tensor *dst) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){0}, UN_SIN);
}
static void cos_kernel(Tensor *src, Tensor *dst) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){0}, UN_COS);
}
static void tg_kernel(Tensor *src, Tensor *dst) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){0}, UN_TG);
}
static void tgh_kernel(Tensor *src, Tensor *dst) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){0}, UN_TGH);
}
static void exp_kernel(Tensor *src, Tensor *dst) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){0}, UN_EXP);
}
static void ln_kernel(Tensor *src, Tensor *dst) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){0}, UN_LN);
}
static void sqr_kernel(Tensor *src, Tensor *dst) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){0}, UN_SQR);
}
static void sqrt_kernel(Tensor *src, Tensor *dst) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){0}, UN_SQRT);
}
static void inv_kernel(Tensor *src, Tensor *dst) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){0}, UN_INV);
}
static void relu_kernel(Tensor *src, Tensor *dst) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){0}, UN_RELU);
}

// --------------CONST KERNELS-----------------------------

static void add_c_kernel(Tensor *src, Tensor *dst, float c) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){.scalar = c}, CONST_ADD);
}
static void sub_c_kernel(Tensor *src, Tensor *dst, float c) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){.scalar = c}, CONST_SUB);
}
static void mul_c_kernel(Tensor *src, Tensor *dst, float c) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){.scalar = c}, CONST_MUL);
}
static void div_c_kernel(Tensor *src, Tensor *dst, float c) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){.scalar = c}, CONST_DIV);
}
static void lt_c_kernel(Tensor *src, Tensor *dst, float c) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){.scalar = c}, CONST_LT);
}
static void le_c_kernel(Tensor *src, Tensor *dst, float c) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){.scalar = c}, CONST_LE);
}
static void eq_c_kernel(Tensor *src, Tensor *dst, float c) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){.scalar = c}, CONST_EQ);
}
static void ne_c_kernel(Tensor *src, Tensor *dst, float c) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){.scalar = c}, CONST_NE);
}
static void ge_c_kernel(Tensor *src, Tensor *dst, float c) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){.scalar = c}, CONST_GE);
}
static void gt_c_kernel(Tensor *src, Tensor *dst, float c) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){.scalar = c}, CONST_GT);
}
static void pow_c_kernel(Tensor *src, Tensor *dst, float c) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){.scalar = c}, CONST_POW);
}
static void log_c_kernel(Tensor *src, Tensor *dst, float c) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){.scalar = c}, CONST_LOG);
}

// -----------------REDUCE KERNELS

static void sum_kernel(Tensor *src, Tensor *dst) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){0}, RD_SUM);
}
static void min_kernel(Tensor *src, Tensor *dst) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){0}, RD_MIN);
}
static void max_kernel(Tensor *src, Tensor *dst) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){0}, RD_MAX);
}
static void prod_kernel(Tensor *src, Tensor *dst) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){0}, RD_PROD);
}

// ------------OTHER KERNELS-----------------------
static void sum_prod_kernel(Tensor **items, int_t n_items) {
    traversal_kernel(items, n_items, (OpCtx){.N = n_items - 1}, SUM_PROD);
}

static void dropout(Tensor *src, Tensor *dst, float p) {
    traversal_kernel((Tensor *[]){src, dst}, 2, (OpCtx){.dropout = {.p = p, .inv_p = 1.f / p}},
                     DROPOUT);
}

static void softmax(Tensor *src, Tensor *dst, Tensor *mx, Tensor *sum) {
    traversal_kernel((Tensor *[]){src, sum, mx}, 3, (OpCtx){0}, _SFTMX_1);
    traversal_kernel((Tensor *[]){src, dst, mx, sum}, 4, (OpCtx){0}, _SFTMX_2);
}

static void mse_kernel(Tensor *pred, Tensor *target, Tensor *dst, int_t N) {
    traversal_kernel((Tensor *[]){pred, target, dst}, 3, (OpCtx){.scalar = 1.f / N}, MSE);
}

// ---------------BACKWARD BINARY KERNELS---------------------------

static void back_add_kernel(Tensor *in_grad, Tensor *out_grad) {
    traversal_kernel((Tensor *[]){in_grad, out_grad}, 2, (OpCtx){0}, BW_ADD);
}
static void back_neg_kernel(Tensor *in_grad, Tensor *out_grad) {
    traversal_kernel((Tensor *[]){in_grad, out_grad}, 2, (OpCtx){0}, BW_NEG);
}
static void back_mul_kernel(Tensor *in_grad, Tensor *out_grad, Tensor *left) {
    traversal_kernel((Tensor *[]){in_grad, out_grad, left}, 3, (OpCtx){0}, BW_MUL);
}
static void back_l_div_kernel(Tensor *in_grad, Tensor *out_grad, Tensor *right) {
    traversal_kernel((Tensor *[]){in_grad, out_grad, right}, 3, (OpCtx){0}, BW_L_DIV);
}
static void back_r_div_kernel(Tensor *in_grad, Tensor *out_grad, Tensor *right, Tensor *out) {
    traversal_kernel((Tensor *[]){in_grad, out_grad, right, out}, 4, (OpCtx){0}, BW_R_DIV);
}
// ----------------BACKWARD UNARY KERNELS-------------------------------

static void back_sin_kernel(Tensor *in_grad, Tensor *out_grad, Tensor *input) {
    traversal_kernel((Tensor *[]){in_grad, out_grad, input}, 3, (OpCtx){0}, BW_SIN);
}
static void back_cos_kernel(Tensor *in_grad, Tensor *out_grad, Tensor *input) {
    traversal_kernel((Tensor *[]){in_grad, out_grad, input}, 3, (OpCtx){0}, BW_COS);
}
static void back_tg_kernel(Tensor *in_grad, Tensor *out_grad, Tensor *output) {
    traversal_kernel((Tensor *[]){in_grad, out_grad, output}, 3, (OpCtx){0}, BW_TG);
}
static void back_tgh_kernel(Tensor *in_grad, Tensor *out_grad, Tensor *output) {
    traversal_kernel((Tensor *[]){in_grad, out_grad, output}, 3, (OpCtx){0}, BW_TGH);
}
static void back_exp_kernel(Tensor *in_grad, Tensor *out_grad, Tensor *output) {
    traversal_kernel((Tensor *[]){in_grad, out_grad, output}, 3, (OpCtx){0}, BW_EXP);
}
static void back_ln_kernel(Tensor *in_grad, Tensor *out_grad, Tensor *input) {
    traversal_kernel((Tensor *[]){in_grad, out_grad, input}, 3, (OpCtx){0}, BW_LN);
}
static void back_sqr_kernel(Tensor *in_grad, Tensor *out_grad, Tensor *input) {
    traversal_kernel((Tensor *[]){in_grad, out_grad, input}, 3, (OpCtx){0}, BW_SQR);
}
static void back_sqrt_kernel(Tensor *in_grad, Tensor *out_grad, Tensor *output) {
    traversal_kernel((Tensor *[]){in_grad, out_grad, output}, 3, (OpCtx){0}, BW_SQRT);
}
static void back_relu_kernel(Tensor *in_grad, Tensor *out_grad, Tensor *input) {
    traversal_kernel((Tensor *[]){in_grad, out_grad, input}, 3, (OpCtx){0}, BW_RELU);
}
static void back_inv_kernel(Tensor *in_grad, Tensor *out_grad, Tensor *output) {
    traversal_kernel((Tensor *[]){in_grad, out_grad, output}, 3, (OpCtx){0}, BW_INV);
}
// ----------------------BACKWARD CONST------------------------------------------
static void back_c_mul_kernel(Tensor *in_grad, Tensor *out_grad, float c) {
    traversal_kernel((Tensor *[]){in_grad, out_grad}, 2, (OpCtx){.scalar = c}, BW_C_MUL);
}
static void back_c_pow_kernel(Tensor *in_grad, Tensor *out_grad, Tensor *input, float c) {
    traversal_kernel((Tensor *[]){in_grad, out_grad, input}, 3, (OpCtx){.scalar = c}, BW_C_POW);
}
static void back_c_log_kernel(Tensor *in_grad, Tensor *out_grad, Tensor *input, float c) {
    traversal_kernel((Tensor *[]){in_grad, out_grad, input}, 3, (OpCtx){.scalar = c}, BW_C_LOG);
}
// ---------------------BACKWARD OTHERS-------------------------
static void back_dropout(Tensor *in_grad, Tensor *out_grad, float p) {
    traversal_kernel((Tensor *[]){in_grad, out_grad}, 2,
                     (OpCtx){.dropout = {.p = p, .inv_p = 1 / (1 - p)}}, BW_DROPOUT);
}
static void back_softmax(Tensor *in_grad, Tensor *out_grad, Tensor *out, Tensor *sum) {
    traversal_kernel((Tensor *[]){sum, out_grad, out}, 3, (OpCtx){.N = 2}, SUM_PROD);
    traversal_kernel((Tensor *[]){in_grad, out_grad, out, sum}, 4, (OpCtx){0}, BW_SOFTMAX);
}
static void back_mse(Tensor *in_grad, Tensor *out_grad, Tensor *pred, Tensor *target, int_t N) {
    traversal_kernel((Tensor *[]){in_grad, out_grad, pred, target}, 4, (OpCtx){.scalar = 1.f / N},
                     BW_MSE);
}
//----------------------MATMUL KERNEL------------------------------------------

static void matmul_2d(int_t M, int_t N, int_t K, float *a_data, float *b_data, float *c_data,
                      int_t *a_st, int_t *b_st, int_t *c_st) {
    float sum;
    for (int_t i = 0; i < M; ++i)
        for (int_t j = 0; j < K; ++j) {
            sum = 0.f;
            for (int_t k = 0; k < N; ++k) {
                sum += a_data[i * a_st[0] + k * a_st[1]] * b_data[k * b_st[0] + j * b_st[1]];
            }
            c_data[i * c_st[0] + j * c_st[1]] += sum;
        }
}

void matmul_kernel(Tensor *a, Tensor *b, Tensor *c) {
    int_t M = a->shape[a->ndims - 2], N = a->shape[a->ndims - 1], K = b->shape[b->ndims - 1];
    int_t batches = 1, batch_ndims = c->ndims - 2, a_st[2], b_st[2], c_st[2], inds[MAX_DIM] = {0};
    for (int_t i = 0; i < a->ndims - 2; ++i)
        batches *= a->shape[i];

    float *restrict a_data = a->storage->data + a->offset,
                    *restrict b_data = b->storage->data + b->offset,
                    *restrict c_data = c->storage->data + c->offset;

    memcpy(a_st, a->strides + a->ndims - 2, 2 * sizeof(int_t));
    memcpy(b_st, b->strides + b->ndims - 2, 2 * sizeof(int_t));
    memcpy(c_st, c->strides + c->ndims - 2, 2 * sizeof(int_t));

    for (int_t i = 0; i < batches; ++i) {
        matmul_2d(M, N, K, a_data, b_data, c_data, a_st, b_st, c_st);

        for (int_t i = batch_ndims - 1; i >= 0; --i) {
            inds[i]++;
            if (inds[i] < c->shape[i]) {
                a_data += a->strides[i], b_data += b->strides[i], c_data += c->strides[i];
                break;
            } else {
                inds[i] = 0;
                a_data -= a->strides[i] * (a->shape[i] - 1);
                b_data -= b->strides[i] * (b->shape[i] - 1);
                c_data -= c->strides[i] * (c->shape[i] - 1);
            }
        }
    }
}

// ----------------------------------OPTIM KERNELS-------------------------------------------------

void adam_step_kernel(Tensor *w, Tensor *m, Tensor *v, float lr, float beta1, float beta2,
                      float eps, float weight_decay, int_t step) {
    float *dw = w->storage->data + w->offset;
    float *dg = w->grad->storage->data + w->grad->offset;
    float *dm = m->storage->data + m->offset;
    float *dv = v->storage->data + v->offset;

    float bias_corr1 = 1.0f - powf(beta1, (float)step);
    float bias_corr2 = 1.0f - powf(beta2, (float)step);
    float step_size = lr / bias_corr1;

    for (int_t i = 0; i < w->numel; ++i) {
        float grad = dg[i];
        // Weight Decay (L2 регуляризация)
        if (weight_decay != 0) grad += weight_decay * dw[i];
        // Обновление моментов
        dm[i] = beta1 * dm[i] + (1.0f - beta1) * grad;
        dv[i] = beta2 * dv[i] + (1.0f - beta2) * grad * grad;

        float denom = sqrtf(dv[i] / bias_corr2) + eps;
        dw[i] -= step_size * (dm[i] / denom);
    }
}

// -----------------------------------NORMALIZATION KERNELS-------------------------------------

void norm_kernel(Tensor *x, Tensor *dst, Tensor *sum, Tensor *sum_sq, float eps, int_t N) {
    float *data_x = x->storage->data, *data_dst = dst->storage->data;
    float *data_sum = sum->storage->data, *data_sq = sum_sq->storage->data;

    int_t coords[MAX_DIM] = {0};
    int_t off_x = x->offset, off_dst = dst->offset, off_stat = sum->offset;
    int_t dim_diff = x->ndims - sum->ndims;

    // Кэшируем значения для текущей группы, чтобы не считать sqrt в каждой итерации
    float cached_mean = data_sum[off_stat] / N;
    float cached_inv_std =
        1.0f / sqrtf((data_sq[off_stat] / N) - (cached_mean * cached_mean) + eps);
    int_t last_stat_off = off_stat;

    for (int_t i = 0; i < x->numel; ++i) {
        // Если смещение по статистикам изменилось — пересчитываем константы
        if (off_stat != last_stat_off) {
            cached_mean = data_sum[off_stat] / N;
            cached_inv_std =
                1.0f / sqrtf((data_sq[off_stat] / N) - (cached_mean * cached_mean) + eps);
            last_stat_off = off_stat;
        }

        data_dst[off_dst] = (data_x[off_x] - cached_mean) * cached_inv_std;

        for (int_t d = x->ndims - 1; d >= 0; --d) {
            int_t dst_d = d - dim_diff;
            if (++coords[d] < x->shape[d]) {
                off_x += x->strides[d];
                if (dst_d >= 0 && sum->shape[dst_d] > 1) off_stat += sum->strides[dst_d];

                break;
            }
            coords[d] = 0;
            off_x -= (x->shape[d] - 1) * x->strides[d];
            if (dst_d >= 0 && sum->shape[dst_d] > 1)
                off_stat -= (sum->shape[dst_d] - 1) * sum->strides[dst_d];
        }
    }
}

BACKEND CPU_NAIVE = {
    //--------BINARY----------
    .add_kernel = &add_kernel,
    .sub_kernel = &sub_kernel,
    .mul_kernel = &mul_kernel,
    .div_kernel = &div_kernel,
    .lt_kernel = &lt_kernel,
    .le_kernel = &le_kernel,
    .eq_kernel = &eq_kernel,
    .ne_kernel = &ne_kernel,
    .ge_kernel = &ge_kernel,
    .gt_kernel = &gt_kernel,
    .assign_kernel = &assign_kernel,
    //--------UNARY-------------
    .neg_kernel = &neg_kernel,
    .sin_kernel = &sin_kernel,
    .cos_kernel = &cos_kernel,
    .tg_kernel = &tg_kernel,
    .tgh_kernel = &tgh_kernel,
    .exp_kernel = &exp_kernel,
    .ln_kernel = &ln_kernel,
    .relu_kernel = &relu_kernel,
    .sqr_kernel = &sqr_kernel,
    .sqrt_kernel = &sqrt_kernel,
    // ---------CONST-------------
    .add_c_kernel = &add_c_kernel,
    .sub_c_kernel = &sub_c_kernel,
    .mul_c_kernel = &mul_c_kernel,
    .div_c_kernel = &div_c_kernel,
    .pow_c_kernel = &pow_c_kernel,
    .log_c_kernel = &log_c_kernel,
    .lt_c_kernel = &lt_c_kernel,
    .le_c_kernel = &le_c_kernel,
    .eq_c_kernel = &eq_c_kernel,
    .ne_c_kernel = &ne_c_kernel,
    .ge_c_kernel = &gt_c_kernel,
    .gt_c_kernel = &gt_c_kernel,
    // --------REDUCE-------------
    .sum_kernel = &sum_kernel,
    .min_kernel = &min_kernel,
    .max_kernel = &max_kernel,
    // .multi_reduce_kernel = &multi_reduce_kernel,
    // --------MATMUL------------
    .matmul_kernel = &matmul_kernel,
    //------------FILL----------
    .fill_from_array_kernel = &fill_from_array_kernel,
    .fill_value_kernel = &fill_value_kernel,
    .fill_range_kernel = &fill_range_kernel,
    .fill_random_uniform = &fill_random_uniform,
    .fill_random_normal = &fill_random_normal,
    // ---------RANDOM----------
    .set_seed = &set_seed,
    .get_seed = &get_seed,
    .dropout = &dropout,
    .softmax = &softmax,
    .sum_prod_kernel = &sum_prod_kernel,
    // ---------OPTIM-----------
    .adam_step_kernel = &adam_step_kernel,
    // ---------NORM----------
    .norm_kernel = &norm_kernel,

    // --------BACKWARD BINARY---------
    .back_add_kernel = &back_add_kernel,
    .back_neg_kernel = &back_neg_kernel,
    .back_mul_kernel = &back_mul_kernel,
    .back_l_div_kernel = &back_l_div_kernel,
    .back_r_div_kernel = &back_r_div_kernel,
    // -------BACKWARD UNARY----------
    .back_sin_kernel = &back_sin_kernel,
    .back_cos_kernel = &back_cos_kernel,
    .back_tg_kernel = &back_tg_kernel,
    .back_tgh_kernel = &back_tgh_kernel,
    .back_ln_kernel = &back_ln_kernel,
    .back_exp_kernel = &back_exp_kernel,
    .back_sqr_kernel = &back_sqr_kernel,
    .back_sqrt_kernel = &back_sqrt_kernel,
    .back_inv_kernel = &back_inv_kernel,
    .back_relu_kernel = &back_relu_kernel,
    // ----BACKWARD CONST------------
    .back_c_mul_kernel = &back_c_mul_kernel,
    .back_c_log_kernel = &back_c_log_kernel,
    .back_c_pow_kernel = &back_c_pow_kernel,
    // -------BACKWARD OTHERS--------------
    .back_dropout = &back_dropout,
    .back_softmax = &back_softmax

};
