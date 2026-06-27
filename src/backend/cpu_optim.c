#include "../core/tensor.h"
#include "../utils/utils.h"
#include "backend.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

// ---------------MY MATH-----------------
// ----------------MY FAST MATH----------------
static FORCE_INLINE float mlog(float x) {
    union {
        float f;
        uint32_t i;
    } vx = {x};
    float exp = (float)(vx.i >> 23) - 127.0f;
    vx.i = (vx.i & 0x7FFFFF) | 0x3f800000;
    float m = vx.f;
    float res = -3.0400429f + (6.1129976f + (-3.3420937f + 0.92383245f * m) * m) * m;
    return (res + exp) * 0.69314718f;
}
static FORCE_INLINE float msin(float x) {
    float x2 = x * x;
    return x * (1.0f + x2 * (-0.16666668f + x2 * (0.0083328241f + x2 * -0.00019587841f)));
}

static FORCE_INLINE float mcos(float x) {
    float x2 = x * x;
    return 1.0f + x2 * (-0.49999999f + x2 * (0.04166664f + x2 * -0.0013888397f));
}
static FORCE_INLINE float mexp(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) return 3.4e38f;

    float z = x * 1.44269504f; // x * log2(e)
    float fn = floorf(z + 0.5f);
    float f = z - fn;
    float res = 1.0f + f * (0.6931472f + f * (0.2402265f + f * (0.0555041f + f * 0.0096181f)));
    union {
        float f;
        uint32_t i;
    } v;
    v.i = ((int32_t)fn + 127) << 23;
    return res * v.f;
}
static FORCE_INLINE float msqrt(float x) { return x / sqrtf(x); }

// -------------------- RANDOM ----------------------------------
#define RS 16

static THREADY uint32_t SEED = 1234567890U;

static FORCE_INLINE uint32_t xor_gen(uint32_t *seed) {
    uint32_t x = *seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *seed = x;
}

static FORCE_INLINE float next_float(uint32_t *seed) {
    return xor_gen(seed) * (1.0f / 4294967296.0f) + 1e-10f;
}
static FORCE_INLINE void thread_seed(uint32_t *seeds, int_t tid) {
    for (int_t i = 0; i < RS; ++i)
        seeds[i] = SEED + 1331 * tid + i;
}

static void randu(float *RE d, int_t n, float min, float max) {
#pragma omp parallel
    {
        int_t tid = omp_get_thread_num(), n_th = omp_get_num_threads();
        int_t start = (n * tid) / n_th, end = (n * (tid + 1)) / n_th;
        uint32_t seeds[RS];
        thread_seed(seeds, tid);
#pragma omp simd
        for (int_t i = start; i < end; ++i) {
            d[i] = min + (max - min) * next_float(&seeds[i % RS]);
        }
    }
}
static void randn(float *RE d, int_t n, float mean, float std) {
#pragma omp parallel
    {
        int tid = omp_get_thread_num(), n_th = omp_get_num_threads();
        int start = (n * tid) / n_th, end = (n * (tid + 1)) / n_th;
        int len = end - start;
        int sz = 16;
        uint32_t seeds[sz];
        thread_seed(seeds, tid);
        int v_len = len & ~1;
#pragma omp simd
        for (int i = 0; i < v_len; i += 2) {
            float u1 = next_float(&seeds[i % 16]);
            float u2 = next_float(&seeds[(i + 1) % 16]);
            float r = msqrt(-2.0f * mlog(u1));
            float theta = 6.2831853f * u2 - 3.1415926f;
            d[start + i] = r * mcos(theta) * std + mean;
            d[start + i + 1] = r * msin(theta) * std + mean;
        }
        if (v_len < len) {
            float u1 = next_float(&seeds[0]);
            float u2 = next_float(&seeds[1]);
            d[start + v_len] = msqrt(-2.0f * mlog(u1)) * mcos(6.28f * u2) * std + mean;
        }
    }
}

// ---------------------------------------------------------------------

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

static FORCE_INLINE void iter_step(TrenIter *it, const Tensor **items, int_t n_items,
                                   int_t right_dim) {
    for (int_t d = right_dim; d >= 0; --d) {
        if (++(it->coords[d]) < items[0]->shape[d]) {
            for (int_t i = 0; i < n_items; ++i)
                it->offsets[i] += items[i]->strides[d];
            break;
        }
        it->coords[d] = 0;
        for (int_t i = 0; i < n_items; ++i)
            it->offsets[i] -= (items[i]->shape[d] - 1) * items[i]->strides[d];
    }
}

void iter_set(TrenIter *it, Tensor **items, int_t n_items, int_t idx, int_t right_dim) {
    int_t remaining = idx;
    for (int_t d = right_dim; d >= 0; --d) {
        int_t coord = remaining % items[0]->shape[d];
        remaining /= items[0]->shape[d];
        it->coords[d] = coord;
        for (int_t t = 0; t < n_items; ++t) {
            it->offsets[t] += coord * items[t]->strides[d];
        }
    }
}

static FORCE_INLINE void red_multi(float **d, OpCtx ctx) {
    if (ctx.red.mask & RED_SUM) *d[RD_SM] += *d[0];
    if (ctx.red.mask & RED_MEAN) *d[RD_MEN] += *d[0] * ctx.red.scalar;
    if (ctx.red.mask & RED_SUM_SQ) *d[RD_SM_SQ] += *d[0] * *d[0];
    if (ctx.red.mask & RED_SQ_MEAN) *d[RD_SQ_MEAN] += *d[0] * ctx.red.scalar;
    if (ctx.red.mask & RED_MAX) *d[RD_MX] = fmaxf(*d[RD_MX], *d[0]);
    if (ctx.red.mask & RED_MIN) *d[RD_MN] = fminf(*d[RD_MN], *d[0]);
    if (ctx.red.mask & RED_PROD) *d[RD_PRD] *= *d[0];
}

static FORCE_INLINE void red_multi_tail(float **tmp, float **acc, OpCtx ctx) {
    if (ctx.red.mask & RED_SUM) *acc[RD_SM] += *tmp[RD_SM];
    if (ctx.red.mask & RED_MEAN) *acc[RD_MEN] += *tmp[RD_MEN] * ctx.red.scalar;
    if (ctx.red.mask & RED_SUM_SQ) *acc[RD_SM_SQ] += *tmp[RD_SM_SQ] * *tmp[RD_SM_SQ];
    if (ctx.red.mask & RED_SQ_MEAN) *acc[RD_SQ_MEAN] += *tmp[RD_SQ_MEAN] * ctx.red.scalar;
    if (ctx.red.mask & RED_MAX) *acc[RD_MX] = fmaxf(*acc[RD_MX], *tmp[RD_MX]);
    if (ctx.red.mask & RED_MAX) *acc[RD_MN] = fminf(*acc[RD_MN], *tmp[RD_MN]);
    if (ctx.red.mask & RED_MAX) *acc[RD_PRD] *= *tmp[RD_PRD];
}

static FORCE_INLINE void sftmx_1(float **d) {
    if (*d[0] > *d[2]) {
        *d[1] = *d[1] * mexp(*d[2] - *d[0]) + 1.f;
        *d[2] = *d[0];
    } else {
        *d[1] += mexp(*d[0] - *d[2]);
    }
}
static FORCE_INLINE void sum_prod(float **d, OpCtx ctx) {
    float el = 1.f;
    for (int_t i = 0; i < ctx.N; ++i)
        el *= *d[i + 1];
    *d[0] += el;
}

static void dropout_kernel(Tensor *out, Tensor *in, float p, bool back) {
    Tensor *items[] = {out, in};
    float *RE outd = out->storage->data + out->offset, *RE ind = in->storage->data + in->offset;
    float i_p = 1 / (1 - p);
    collapse_dims(items, 2, out->ndims);
    if (!check_last_stride(items, 2)) {
        // TODO: naive call
        return;
    }
    int_t cols = out->shape[out->ndims - 1], rows = out->numel / out->shape[out->ndims - 1];
#pragma omp parallel
    {
        int_t tid = omp_get_thread_num(), n_th = omp_get_num_threads();
        uint32_t seeds[RS];
        thread_seed(seeds, tid);
        if (out->ndims == 1) {
            int_t start = (cols * tid) / n_th, end = (cols * (tid + 1)) / n_th;
            if (back) {
#pragma omp simd
                for (int_t i = start; i < end; ++i)
                    outd[i] += next_float(&seeds[i % RS]) < p ? i_p * ind[i] : 0.f;
            } else {
#pragma omp simd
                for (int_t i = start; i < end; ++i)
                    outd[i] = next_float(&seeds[i % RS]) < p ? i_p * ind[i] : 0.f;
            }
        } else {
            TrenIter it = iter_new(items, 2);
            int_t start = (rows * tid) / n_th, end = (rows * (tid + 1)) / n_th;
            for (int_t i = start; i < end; ++i) {
                float *out_t = outd + it.offsets[0], *in_t = ind + it.offsets[1];
                if (back) {
#pragma omp simd
                    for (int_t j = 0; j < cols; ++j)
                        out_t[j] += next_float(&seeds[j % RS]) < p ? i_p * in_t[j] : 0.f;
                } else {
#pragma omp simd
                    for (int_t j = 0; j < cols; ++j)
                        out_t[j] = next_float(&seeds[j % RS]) < p ? i_p * in_t[j] : 0.f;
                }
                iter_step(&it, items, 2, out->ndims - 2);
            }
        }
    }
}

static FORCE_INLINE void apply_op(float **d, OpType op, OpCtx ctx) {
    switch (op) {

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
    case RD_PROD   : *d[0] *= *d[0]; break;
    case RD_SUM    : *d[0] += *d[0]; break;
    case RD_MAX    : *d[0] = fmaxf(*d[1], *d[0]); break;
    case RD_MIN    : *d[1] = fminf(*d[1], *d[0]); break;
    case RD_SUM_SQ : *d[0] += *d[1] * *d[1]; break;
    case RD_MULTI  : red_multi(d, ctx); break;
    // --------------OTHER UTILS---------------------
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

    case BW_SOFTMAX: *d[0] = *d[2] * (*d[1] - *d[3]); break;
    case BW_MSE    : *d[0] += *d[1] * 2.f * (*d[2] - *d[3]) * ctx.scalar; break;
    default        : break;
    }
}

static FORCE_INLINE void double_1d(float *RE out, float *RE in, int_t cols, OpType op, OpCtx ctx,
                                   bool paral) {
#pragma omp parallel for simd if (cols > 10000 && paral)
    for (int_t i = 0; i < cols; ++i) {
        apply_op((float *[]){&out[i], &in[i]}, op, ctx);
    }
}

static FORCE_INLINE void double_nd(Tensor **items, int_t rows, int_t cols, OpType op, OpCtx ctx) {
    int_t ndims = items[0]->ndims;
    float *out = items[0]->storage->data + items[0]->offset,
          *in = items[1]->storage->data + items[1]->offset;
#pragma omp parallel
    {
        TrenIter it = iter_new(items, 2);
        int tid = omp_get_thread_num(), n_threads = omp_get_num_threads();
        int_t start = (rows * tid) / n_threads, end = (rows * (tid + 1)) / n_threads;
        iter_set(&it, items, 2, start, ndims - 2);
        for (int_t i = start; i < end; ++i) {
            double_1d(out + it.offsets[0], in + it.offsets[1], cols, op, ctx, false);
            iter_step(&it, items, 2, ndims - 2);
        }
    }
}

static FORCE_INLINE void triple_1d_sv(float *RE out, float *RE left, float *RE right, int_t cols,
                                      OpType op, OpCtx ctx, bool paral) {
#pragma omp parallel for simd if (cols > 10000 && paral)
    for (int_t i = 0; i < cols; ++i)
        apply_op((float *[]){&out[i], left, &right[i]}, op, ctx);
}

static FORCE_INLINE void triple_1d_vs(float *RE out, float *RE left, float *RE right, int_t cols,
                                      OpType op, OpCtx ctx, bool paral) {
#pragma omp parallel for simd if (cols > 10000 && paral)
    for (int_t i = 0; i < cols; ++i)
        apply_op((float *[]){&out[i], &left[i], right}, op, ctx);
}

static FORCE_INLINE void triple_1d_vv(float *RE out, float *RE left, float *RE right, int_t cols,
                                      OpType op, OpCtx ctx, bool paral) {
#pragma omp parallel for simd if (cols > 10000 && paral)
    for (int_t i = 0; i < cols; ++i)
        apply_op((float *[]){&out[i], &left[i], &right[i]}, op, ctx);
}

static FORCE_INLINE void triple_nd(Tensor **items, int_t rows, int_t cols, OpType op, OpCtx ctx,
                                   int_t mask) {
    int_t ndims = items[0]->ndims;
    float *RE out = items[0]->storage->data + items[0]->offset,
              *RE left = items[1]->storage->data + items[1]->offset,
              *RE right = items[2]->storage->data + items[2]->offset;
#pragma omp parallel
    {
        TrenIter it = iter_new(items, 2);
        int tid = omp_get_thread_num(), n_threads = omp_get_num_threads();
        int_t start = (rows * tid) / n_threads, end = (rows * (tid + 1)) / n_threads;
        iter_set(&it, items, 2, start, ndims - 2);
        for (int_t i = start; i < end; ++i) {
            float *out_t = out + it.offsets[0], *left_t = left + it.offsets[1],
                  *right_t = right + it.offsets[2];
            switch (mask) {
            case 1: triple_1d_sv(out_t, left_t, right_t, cols, op, ctx, true); break;
            case 2: triple_1d_vs(out_t, left_t, right_t, cols, op, ctx, true); break;
            case 3: triple_1d_vv(out_t, left_t, right_t, cols, op, ctx, true); break;
            }
            iter_step(&it, items, 2, ndims - 2);
        }
    }
}

static FORCE_INLINE void universal_entrance(Tensor **items, int_t n_items, OpType op, OpCtx ctx) {
    collapse_dims(items, n_items, items[0]->ndims);
    if (!check_last_stride(items, n_items)) {
        // TODO: naive call
        return;
    }
    float **d[MAX_OPERANDS];
    for (int_t i = 0; i < n_items; ++i)
        d[i] = items[i]->storage->data + items[i]->offset;
    int_t l_str[MAX_OPERANDS], ndims = items[0]->ndims;
    for (int_t i = 0; i < n_items; ++i)
        l_str[i] = items[i]->strides[ndims];
    int_t cols = items[0]->shape[ndims - 1], rows = items[0]->numel / cols;
    switch (n_items) {
    case 2: {
        if (ndims == 1) double_1d(d[0], d[1], cols, op, ctx, true);
        else double_nd(items, rows, cols, op, ctx);
        break;
    }
    case 3: {
        const int_t mask = (l_str[1] == 1) | (l_str[2] == 1) << 1;
        if (ndims == 1) {
            switch (mask) {
            case 1: triple_1d_sv(d[0], d[1], d[2], cols, op, ctx, true); break;
            case 2: triple_1d_vs(d[0], d[1], d[2], cols, op, ctx, true); break;
            case 3: triple_1d_vv(d[0], d[1], d[2], cols, op, ctx, true); break;
            }
        } else {
            triple_nd(items, rows, cols, op, ctx, mask);
        }
        break;
    case 4: {
        break;
    }
    }
    }
}

static void add_kernel(Tensor *out, Tensor *left, Tensor *right) {
    universal_entrance((Tensor *[]){out, left, right}, 3, BIN_ADD, (OpCtx){0});
}
static void sub_kernel(Tensor *out, Tensor *left, Tensor *right) {
    universal_entrance((Tensor *[]){out, left, right}, 3, BIN_SUB, (OpCtx){0});
}
static void mul_kernel(Tensor *out, Tensor *left, Tensor *right) {
    universal_entrance((Tensor *[]){out, left, right}, 3, BIN_MUL, (OpCtx){0});
}
static void div_kernel(Tensor *out, Tensor *left, Tensor *right) {
    universal_entrance((Tensor *[]){out, left, right}, 3, BIN_DIV, (OpCtx){0});
}
static void lt_kernel(Tensor *out, Tensor *left, Tensor *right) {
    universal_entrance((Tensor *[]){out, left, right}, 3, BIN_LT, (OpCtx){0});
}
static void le_kernel(Tensor *out, Tensor *left, Tensor *right) {
    universal_entrance((Tensor *[]){out, left, right}, 3, BIN_LE, (OpCtx){0});
}
static void eq_kernel(Tensor *out, Tensor *left, Tensor *right) {
    universal_entrance((Tensor *[]){out, left, right}, 3, BIN_EQ, (OpCtx){0});
}
static void ne_kernel(Tensor *out, Tensor *left, Tensor *right) {
    universal_entrance((Tensor *[]){out, left, right}, 3, BIN_NE, (OpCtx){0});
}
static void ge_kernel(Tensor *out, Tensor *left, Tensor *right) {
    universal_entrance((Tensor *[]){out, left, right}, 3, BIN_GE, (OpCtx){0});
}
static void gt_kernel(Tensor *out, Tensor *left, Tensor *right) {
    universal_entrance((Tensor *[]){out, left, right}, 3, BIN_GT, (OpCtx){0});
}

static FORCE_INLINE void red_1d(float **d, int_t cols, OpCtx ctx) {
#pragma omp parallel
    {

        float *in = d[0];
        float sum = ctx.red.mask & RED_SUM ? *d[RD_SM] : 0;
        float mean = ctx.red.mask & RED_MEAN ? *d[RD_MEN] : 0;
        float sq = ctx.red.mask & RED_SUM_SQ ? *d[RD_SM_SQ] : 0;
        float sq_mn = ctx.red.mask & RED_SQ_MEAN ? *d[RD_SQ_MEAN] : 0;
        float mx = ctx.red.mask & RED_MAX ? *d[RD_MX] : 0;
        float mn = ctx.red.mask & RED_MIN ? *d[RD_MN] : 0;
        float prod = ctx.red.mask & RED_PROD ? *d[RD_PRD] : 0;
        int_t tid = omp_get_thread_num(), n_th = omp_get_num_threads();
        int_t start = (tid * cols) / n_th, end = ((tid + 1) * cols) / n_th;
#pragma omp simd
        for (int_t i = start; i < end; ++i) {
            red_multi((float *[]){&in[i], &sum, &mean, &sq, &sq_mn, &mx, &mn, &prod}, ctx);
        }

#pragma omp critical
        {
            red_multi_tail((float *[]){&sum, &mean, &sq, &sq_mn, &mx, &mn, &prod}, d + 1, ctx);
        }
    }
}

static FORCE_INLINE void red_nd_last(Tensor **items, int_t n_items, int_t rows, int_t cols,
                                     OpCtx ctx, bool paral) {
#pragma omp parallel if (rows > 8)
    {
        TrenIter it = iter_new(items, RD_COUNT + 1);
        int_t tid = omp_get_thread_num(), n_threads = omp_get_num_threads();
        int_t start = (rows * tid) / n_threads, end = (rows * (tid + 1)) / n_threads;
        iter_set(&it, items, 2, start, items[0]->ndims - 2);
        float *d[RD_COUNT] = {NULL};
        for (int_t i = 0; i < RD_COUNT; ++i)
            d[i] = items[i + 1]->storage->data + items[i + 1]->offset;
        float *RE in = items[0]->storage->data + items[0]->offset;
        for (int_t i = start; i < rows; ++i) {
            float *RE in_t = in + it.offsets[0];
            float *RE sum = ctx.red.mask & RED_SUM ? d[RD_SM] + it.offsets[RD_SM] : NULL;
            float *RE mean = ctx.red.mask & RED_MEAN ? d[RD_MEN] + it.offsets[RD_MEN] : NULL;
            float *RE sq = ctx.red.mask & RED_SUM_SQ ? d[RD_SM_SQ] + it.offsets[RD_SM_SQ] : NULL;
            float *RE sq_mn =
                ctx.red.mask & RED_SUM_SQ ? d[RD_SQ_MEAN] + it.offsets[RD_SQ_MEAN] : NULL;
            float *RE mx = ctx.red.mask & RED_MAX ? d[RD_MX] + it.offsets[RD_MX] : NULL;
            float *RE mn = ctx.red.mask & RED_MIN ? d[RD_MN] + it.offsets[RD_MN] : NULL;
            float *RE prod = ctx.red.mask & RED_PROD ? d[RD_PRD] + it.offsets[RD_PRD] : NULL;
            float s1 = *sum, s2 = *mean, s3 = *sq, s4 = *sq_mn, s5 = *mx, s6 = *mn, s7 = *prod;
#pragma omp simd
            for (int_t j = 0; j < cols; ++j)
                red_multi((float *[]){&in_t[j], &s1, &s2, &s3, &s4, &s5, &s6, &s7}, ctx);
            red_multi_tail((float *[]){&s1, &s2, &s3, &s4, &s5, &s6, &s7},
                           (float *[]){sum, mean, sq, sq_mn, mx, mn, prod}, ctx);
            iter_step(&it, items, n_items, items[0]->ndims - 2);
        }
    }
}

static FORCE_INLINE void red_nd_mid(Tensor **items, int_t n_items, int_t outer, int_t mid,
                                    int_t inner, OpCtx ctx) {
#pragma omp paralel
    {
        TrenIter it = iter_new(items, RD_COUNT + 1);
        int_t tid = omp_get_thread_num(), n_threads = omp_get_num_threads();
        int_t start = (outer * tid) / n_threads, end = (outer * (tid + 1)) / n_threads;
        iter_set(&it, items, 2, start, items[0]->ndims - 3);
        float *d[RD_COUNT] = {NULL};
        for (int_t i = 0; i < RD_COUNT; ++i)
            d[i] = items[i + 1]->storage->data + items[i + 1]->offset;
        float *RE in = items[0]->storage->data + items[0]->offset;
        for (int_t i = start; i < end; ++i) {
            float *RE in_t = in + it.offsets[0];
            float *RE sum = ctx.red.mask & RED_SUM ? d[RD_SM] + it.offsets[RD_SM] : NULL;
            float *RE mean = ctx.red.mask & RED_MEAN ? d[RD_MEN] + it.offsets[RD_MEN] : NULL;
            float *RE sq = ctx.red.mask & RED_SUM_SQ ? d[RD_SM_SQ] + it.offsets[RD_SM_SQ] : NULL;
            float *RE sq_mn =
                ctx.red.mask & RED_SUM_SQ ? d[RD_SQ_MEAN] + it.offsets[RD_SQ_MEAN] : NULL;
            float *RE mx = ctx.red.mask & RED_MAX ? d[RD_MX] + it.offsets[RD_MX] : NULL;
            float *RE mn = ctx.red.mask & RED_MIN ? d[RD_MN] + it.offsets[RD_MN] : NULL;
            float *RE prod = ctx.red.mask & RED_PROD ? d[RD_PRD] + it.offsets[RD_PRD] : NULL;
            for (int_t j = 0; j < mid; ++j) {
#pragma omp simd
                for (int_t k = 0; k < inner; ++k) {
                    red_multi((float *[]){&in_t[k], &sum[k], &mean[k], &sq[k], &sq_mn[k], &mx[k],
                                          &mn[k], &prod[k]},
                              ctx);
                }
            }
            in_t += items[0]->strides[items[0]->ndims - 2];
            iter_step(&it, items, n_items, items[0]->ndims - 3);
        }
    }
}
static void red_3d_outer_inner(Tensor **items, int_t outer, int_t mid, int_t inner, OpCtx ctx) {
#pragma omp paralel
    {
        int_t tid = omp_get_thread_num(), n_threads = omp_get_num_threads();
        int_t start = (mid * tid) / n_threads, end = (mid * (tid + 1)) / n_threads;
        if (start < end) {
            float *d[RD_COUNT] = {NULL};
            for (int_t i = 0; i < RD_COUNT; ++i)
                d[i] = items[i + 1]->storage->data + items[i + 1]->offset;
            float *RE in = items[0]->storage->data + items[0]->offset;
            TrenIter it1 = iter_new(items, RD_COUNT);
            iter_set(&it1, items, 2, start, items[0]->ndims - 2);
            for (int_t i = 0; i < outer; ++i) {
                TrenIter it = it1;
                for (int_t j = start; j < end; ++j) {
                    float *RE in_t = in + it.offsets[0];
                    float *RE sum = ctx.red.mask & RED_SUM ? d[RD_SM] + it.offsets[RD_SM] : NULL;
                    float *RE mean =
                        ctx.red.mask & RED_MEAN ? d[RD_MEN] + it.offsets[RD_MEN] : NULL;
                    float *RE sq =
                        ctx.red.mask & RED_SUM_SQ ? d[RD_SM_SQ] + it.offsets[RD_SM_SQ] : NULL;
                    float *RE sq_mn =
                        ctx.red.mask & RED_SUM_SQ ? d[RD_SQ_MEAN] + it.offsets[RD_SQ_MEAN] : NULL;
                    float *RE mx = ctx.red.mask & RED_MAX ? d[RD_MX] + it.offsets[RD_MX] : NULL;
                    float *RE mn = ctx.red.mask & RED_MIN ? d[RD_MN] + it.offsets[RD_MN] : NULL;
                    float *RE prod =
                        ctx.red.mask & RED_PROD ? d[RD_PRD] + it.offsets[RD_PRD] : NULL;
                    float s1 = *sum, s2 = *mean, s3 = *sq, s4 = *sq_mn, s5 = *mx, s6 = *mn,
                          s7 = *prod;
#pragma omp simd
                    for (int_t k = 0; k < inner; ++k) {
                        red_multi((float *[]){&in_t[k], &s1, &s2, &s3, &s4, &s5, &s6, &s7}, ctx);
                    }
                    red_multi_tail((float *[]){&s1, &s2, &s3, &s4, &s5, &s6, &s7},
                                   (float *[]){sum, mean, sq, sq_mn, mx, mn, prod}, ctx);

                    iter_step(&it, items, RD_COUNT, items[0]->ndims - 2);
                }
                iter_step(&it1, items, RD_COUNT, items[0]->ndims - 3);
            }
        }
    }
}

static FORCE_INLINE void reduction_entrance(Tensor **items, int_t n_items, int_t *axes,
                                            int_t n_axes, OpType op, OpCtx ctx) {
    collapse_reduce_ndims(items, n_items, axes, n_axes);
    bool axes[MAX_DIM];
    int_t n_axes, ndims = items[0]->ndims;
    find_reduce_axis(items, n_items);
    int_t inner = items[0]->shape[ndims - 1], out = items[0]->numel / items[0]->shape[ndims - 1];
    if (!check_last_stride(items, n_items)) {
        // TODO: naive call
        return;
    }
    int_t inner = items[0]->shape[ndims - 1];
    int_t mid = axes[ndims - 2] || axes[ndims - 3] ? items[0]->shape[ndims - 2] : 1;
    int_t outer = items[0]->numel / (inner * mid);
    float **d[RD_COUNT + 1] = {[0] = items[0]->storage->data + items[0]->offset};
    for (int_t i = 1; i < n_items; ++i)
        d[i] = ctx.red.mask & 1 << i ? items[i]->storage->data + items[i]->offset : NULL;
    if (ndims == 1) red_1d(d, inner, ctx);
    else if (n_axes == 1 && axes[ndims - 1]) red_nd_last(items, n_items, outer, inner, ctx, true);
    else if (n_axes == 1 && axes[ndims - 2]) red_nd_mid(items, n_items, outer, mid, inner, ctx);
    else if (n_axes == 2 && axes[ndims - 1] && axes[ndims - 3])
        red_3d_outer_inner(items, outer, mid, inner, ctx);
    else {
        // TODO:naive call
        return;
    }
}
BACKEND CPU_OPTIM = {0

};

// #define DISPATCH_OpType(op_var, loop_body)                                                             \
//     switch (op_var) {                                                                              \
//     case BIN_ADD: {                                                                                \
//         const OpType current_op = BIN_ADD;                                                             \
//         loop_body;                                                                                 \
//     } break;                                                                                       \
//     case BIN_MUL: {                                                                                \
//         const OpType current_op = BIN_MUL;                                                             \
//         loop_body;                                                                                 \
//     } break;                                                                                       \
//     case UN_EXP: {                                                                                 \
//         const OpType current_op = UN_EXP;                                                              \
//         loop_body;                                                                                 \
//     } break;                                                                                       \
//     /* ... перечислить нужные здесь операции ... */                                                \
//     default: break;                                                                                \
//     }

// // 2. Используем его в функции
// static FORCE_INLINE void triple_1d_vv(float *RE out, float *RE left,
//                                       float *RE right, int_t in, OpType op, OpCtx ctx,
//                                       bool paral) {
//     // Весь этот блок будет виден Intellisense
//     DISPATCH_OpType(op, {
//         PRAGMA_PARALLEL_SIMD(inner > 10000 && paral);
//         for (int_t i = 0; i < cols; ++i) {
//             // Внутри этого блока current_op — это константа (BIN_ADD и т.д.)
//             // MSVC увидит, что switch внутри apply_op можно полностью удалить (Dead Code
//             // Elimination)
//             apply_op((float *[]){&out[i], &left[i], &right[i]}, current_op, ctx);
//         }
//     });
// }
