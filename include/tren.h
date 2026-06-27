#pragma once
#include "tren_types.h"
#include <stdbool.h>
#include <stdint.h>

typedef struct Tensor Tensor;
#if defined(__clang__)
#define msg "Clang"
#endif
#if defined(__MSV_VER__)
#define msg "MSVC"
#endif

// --------------COMMON OPS----------------------------

Tensor *tren_create(const int_t *shape, int_t ndims, DEVICE d, bool requires_grad);
Tensor *tren_view(Tensor *src, const int_t *new_shape, int_t new_ndims);
Tensor *tren_slice(Tensor *src, Slice *slices, int_t ndims);
Tensor *tren_transpose(Tensor *src, int_t *order, int_t ndims);
void tren_free(Tensor *t);
Tensor *tren_copy(const Tensor *t, DEVICE d);
void tren_print(const Tensor *t);
Tensor *tren_squeeze(Tensor *t);
float tren_get_scalar_value(const Tensor *t);
void tren_assign(Tensor *a, Tensor *b);

// ---------------BINARY OPS-------------------------------

Tensor *tren_add(Tensor *a, Tensor *b, bool in_place);
Tensor *tren_sub(Tensor *a, Tensor *b, bool in_place);
Tensor *tren_mul(Tensor *a, Tensor *b, bool in_place);
Tensor *tren_div(Tensor *a, Tensor *b, bool in_place);
Tensor *tren_lt(Tensor *a, Tensor *b, bool in_place);
Tensor *tren_le(Tensor *a, Tensor *b, bool in_place);
Tensor *tren_eq(Tensor *a, Tensor *b, bool in_place);
Tensor *tren_ne(Tensor *a, Tensor *b, bool in_place);
Tensor *tren_ge(Tensor *a, Tensor *b, bool in_place);
Tensor *tren_gt(Tensor *a, Tensor *b, bool in_place);

// -----------------UNARY OPS-----------------------------

Tensor *tren_neg(Tensor *a, bool in_place);
Tensor *tren_sin(Tensor *a, bool in_place);
Tensor *tren_cos(Tensor *a, bool in_place);
Tensor *tren_tg(Tensor *a, bool in_place);
Tensor *tren_tgh(Tensor *a, bool in_place);
Tensor *tren_exp(Tensor *a, bool in_place);
Tensor *tren_ln(Tensor *a, bool in_place);
Tensor *tren_relu(Tensor *a, bool in_place);
Tensor *tren_sqr(Tensor *a, bool in_place);
Tensor *tren_sqrt(Tensor *a, bool in_place);
Tensor *tren_inv(Tensor *a, bool in_place);

// -----------------CONST OPS----------------------------------

Tensor *tren_add_const(Tensor *a, const float c, bool in_place);
Tensor *tren_sub_const(Tensor *a, const float c, bool in_place);
Tensor *tren_mul_const(Tensor *a, const float c, bool in_place);
Tensor *tren_div_const(Tensor *a, const float c, bool in_place);
Tensor *tren_log_const(Tensor *a, const float c, bool in_place);
Tensor *tren_pow_const(Tensor *a, const float c, bool in_place);
Tensor *tren_lt_const(Tensor *a, const float c, bool in_place);
Tensor *tren_le_const(Tensor *a, const float c, bool in_place);
Tensor *tren_eq_const(Tensor *a, const float c, bool in_place);
Tensor *tren_ne_const(Tensor *a, const float c, bool in_place);
Tensor *tren_ge_const(Tensor *a, const float c, bool in_place);
Tensor *tren_gt_const(Tensor *a, const float c, bool in_place);

// -----------------REDUCE OPS--------------------------

Tensor *tren_sum(Tensor *a, int_t *axes, int_t n_axes);
Tensor *tren_mean(Tensor *a, int_t *axes, int_t n_axes);
Tensor *tren_min(Tensor *a, int_t *axes, int_t n_axes);
Tensor *tren_max(Tensor *a, int_t *axes, int_t n_axes);

// -----------------FILL OPS--------------------------------

void tren_fill_value(Tensor *a, const float c);
void tren_fill_from_array(Tensor *a, const float *arr, int_t numel);
void tren_fill_random_uniform(Tensor *a, float min, float max);
void tren_fill_random_normal(Tensor *a, float mean, float std);
Tensor *tren_range(int_t start, int_t end, int_t step, DEVICE d, bool requires_grad);

// -----------------MATMUL-------------------------------

Tensor *tren_matmul(Tensor *a, Tensor *b);

// ----------------SERVICE-------------------------------

void tren_seed_cpu_random(uint64_t seed);

// ----------------OTHER-----------------------

Tensor *tren_dropout(Tensor *a, float p, bool in_place);
Tensor *tren_softmax(Tensor *a, int_t *axes, int_t n_axes);
Tensor *tren_Softmax_CrossEntropy(Tensor *a, int_t *axes, int_t n_axes);

// -----------------GETTERS----------------------------

const float *tren_get_data(Tensor *t);
const int_t tren_get_ndims(Tensor *t);
const int_t *tren_get_shape(Tensor *t);
const int_t *tren_get_strides(Tensor *t);
const int_t tren_get_numel(Tensor *t);
Tensor *tren_get_grad(Tensor *t);
