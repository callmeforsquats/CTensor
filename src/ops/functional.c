#include "../backend/backend.h"
#include "../core/tensor.h"
#include "../utils/error.h"
#include "../utils/utils.h"
#include "backward.h"
#include "tren.h"
#include <float.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

//---------------------SERVICE--------------------------------------

void tren_seed_cpu_random(uint64_t seed) { CPU_NAIVE.set_seed(seed); }

//------------------------------GETTERS-----------------------------

const float *tren_get_data(Tensor *t) { return t->storage->data; }
const int_t tren_get_ndims(Tensor *t) { return t->ndims; }
const int_t *tren_get_shape(Tensor *t) { return t->shape; };
const int_t *tren_get_strides(Tensor *t) { return t->strides; };
const int_t tren_get_numel(Tensor *t) { return t->numel; };
Tensor *tren_get_grad(Tensor *t) { return t->grad; };

//--------------------------------COMMON OPS-----------------------------------

Tensor *tren_create(const int_t *shape, int_t ndims, DEVICE d, bool requires_grad) {
    TREN_CHECK(ndims >= 0, "ndims cannot be negative");
    TREN_CHECK(check_shape(shape, ndims), "Invalid shape: %s", tuple_str(shape, ndims));
    Tensor *res = tensor_create_empty(shape, ndims, d);
    res->requires_grad = requires_grad;
    return res;
}

Tensor *tren_range(int_t start, int_t end, int_t step, DEVICE d, bool requires_grad) {
    TREN_CHECK(step != 0, "Step cannot be zero");
    Tensor *res = tensor_create_empty((int_t[]){(end - start) / step}, 1, d);
    res->requires_grad = requires_grad;
    GET_BACKEND(d)->fill_range_kernel(res, start, step);
    return res;
}

Tensor *tren_view(Tensor *src, const int_t *new_shape, int_t new_ndims) {
    TREN_CHECK(tensor_is_contigous(src), "Cannot create a view of non-contiguous tensor");
    TREN_CHECK(new_ndims >= 0, "ndims cannot be negative");
    TREN_CHECK(check_shape(new_shape, new_ndims), "Invalid shape: %s",
               tuple_str(new_shape, new_ndims));
    TREN_CHECK(src->numel == calc_numel(new_shape, new_ndims),
               "Numel missmatch: view %s, tensor %s", tuple_str(src->shape, src->ndims),
               tuple_str(new_shape, new_ndims));

    Tensor *res = tensor_view(src, new_shape, new_ndims, false);
    res->requires_grad = src->requires_grad;
    if (src->requires_grad) op_create(res, (Tensor *[]){src}, 1, backward_view, "View", NULL, free);
    return res;
}

Tensor *tren_slice(Tensor *src, Slice *slices, int_t ndims) {
    TREN_CHECK(ndims >= 0, "ndims cannot be negative");
    TREN_CHECK(check_slices(src, slices, ndims), "Invalid slices %s for shape %s",
               slice_str(slices, ndims), tuple_str(src->shape, src->ndims));
    Tensor *res = tensor_slice(src, slices, ndims);
    if (src->requires_grad) {
        SliceCtx *ctx = (SliceCtx *)malloc(sizeof(SliceCtx));
        ctx->n_slices = ndims;
        memcpy(ctx->slices, slices, ndims * sizeof(Slice));
        op_create(res, (Tensor *[]){src}, 1, backward_slice, "Slice", ctx, free);
    }
    return res;
}

Tensor *tren_transpose(Tensor *src, int_t *order, int_t ndims) {
    TREN_CHECK(ndims = src->ndims, "Cannot transpose %lldD tensor with %lld axes", src->ndims,
               ndims);
    Tensor *res = tensor_transpose(src, order, false);
    if (src->requires_grad) {
        TransposeCtx *ctx = (TransposeCtx *)malloc(sizeof(TransposeCtx));
        memcpy(ctx->order, order, ndims * sizeof(int_t));
        op_create(res, (Tensor *[]){src}, 1, backward_transpose, "Transpose", ctx, free);
    }
    return res;
}

void tren_free(Tensor *t) { tensor_free(t); }

Tensor *tren_copy(const Tensor *t, DEVICE d) { return tensor_copy(t, d); }

void tren_print(const Tensor *t) {
    if (t->storage->device == CPU_DEVICE) tensor_print(t);
}

Tensor *tren_squeeze(Tensor *t) { return tensor_squeeze(t, false); }

float tren_get_scalar_value(const Tensor *t) {
    TREN_CHECK(t->ndims == 0 && t->numel == 1, "Cannot get a value of shape %s and %lld numel",
               tuple_str(t->shape, t->ndims), t->numel);
    return t->storage->data[t->offset];
}

void tren_assign(Tensor *dst, Tensor *src) {
    TREN_CHECK(dst->storage->device == src->storage->device, "Device missmatch while assignment");
    TREN_CHECK(tuples_equal(dst->shape, src->shape, dst->ndims, src->ndims),
               "Shape missmatch while assignment: %s , %s", tuple_str(dst->shape, dst->ndims),
               tuple_str(src->shape, src->ndims));
    GET_BACKEND(dst->storage->device)->assign_kernel(src, dst);
}

void tren_fill_batch(Tensor *dst, Tensor *src, int_t *inds, int_t n) {
    TREN_CHECK(src->storage->device == dst->storage->device, "Device missmatch");
    TREN_CHECK(tuples_equal(src->shape + 1, dst->shape + 1, src->ndims - 1, dst->ndims - 1),
               "Shape missmatch of tensor %s and batch %s", tuple_str(src->shape, src->ndims),
               tuple_str(dst->shape, dst->ndims));
    BACKEND *backend = GET_BACKEND(src->storage->device);
    for (int_t i = 0; i < n; ++i) {
        Slice s1 = {.start = inds[i]}, s2 = {.start = i};
        TREN_CHECK(check_slice(&s1, src->shape[0]), "Invalid index %lld/%lld", inds[i], dst->ndims);
        Tensor *src_slice = tensor_slice(src, &s1, 1);
        Tensor *dst_slice = tensor_slice(src, &s2, 1);
        backend->assign_kernel(src_slice, dst_slice);
        tensor_free(src_slice), tensor_free(dst_slice);
    }
}

//------------------------------------BINARY OPS---------------------------------

static void (*bin_fn(int_t op_id, BACKEND *backend))(Tensor *left, Tensor *right, Tensor *res) {
    switch (op_id) {
    case BIN_ADD: return backend->add_kernel;
    case BIN_SUB: return backend->sub_kernel;
    case BIN_MUL: return backend->mul_kernel;
    case BIN_DIV: return backend->div_kernel;
    case BIN_LT : return backend->lt_kernel;
    case BIN_LE : return backend->le_kernel;
    case BIN_EQ : return backend->eq_kernel;
    case BIN_NE : return backend->ne_kernel;
    case BIN_GE : return backend->ge_kernel;
    case BIN_GT : return backend->gt_kernel;
    default     : return NULL;
    }
}
static Tensor *binary_op(Tensor *a, Tensor *b, bool in_place, int_t op_id, back_fun *backward,
                         char *name) {
    TREN_CHECK(a->storage->device == b->storage->device, "Tensors are on different devices");
    TREN_CHECK(!a->requires_grad || !in_place, "In-place %s with grad required not allowed", name);
    int_t ndims = max(a->ndims, b->ndims), shape[MAX_DIM];
    DEVICE device = a->storage->device;
    calc_broadcast_result_shape(a, b, shape, ndims);
    TREN_CHECK(!in_place || tuples_equal(a->shape, shape, a->ndims, ndims),
               "Cannot make in-place %s: operand shape=%s, result shape=%s", name,
               tuple_str(a->shape, a->ndims), tuple_str(shape, ndims));
    Tensor a_cast = make_broadcast_view(a, shape, ndims);
    Tensor b_cast = make_broadcast_view(b, shape, ndims);
    Tensor *c = in_place ? a : tensor_create_empty(shape, ndims, device);
    c->requires_grad = a->requires_grad || b->requires_grad;
    BACKEND *backend = GET_BACKEND(device);
    bin_fn(op_id, backend)(&a_cast, &b_cast, c);
    if (c->requires_grad) {
        op_create(c, (Tensor *[]){a, b}, 2, backward, name, NULL, free);
    }
    return c;
}
Tensor *tren_add(Tensor *a, Tensor *b, bool in_place) {
    return binary_op(a, b, in_place, BIN_ADD, backward_ones, "Add");
}
Tensor *tren_sub(Tensor *a, Tensor *b, bool in_place) {
    return binary_op(a, b, in_place, BIN_SUB, backward_sub, "Sub");
}
Tensor *tren_mul(Tensor *a, Tensor *b, bool in_place) {
    return binary_op(a, b, in_place, BIN_MUL, backward_mul, "Mul");
}
Tensor *tren_div(Tensor *a, Tensor *b, bool in_place) {
    return binary_op(a, b, in_place, BIN_DIV, backward_add, "Div");
}
Tensor *tren_lt(Tensor *a, Tensor *b, bool in_place) {
    return binary_op(a, b, in_place, BIN_LT, backward_zeros, "BIN_LT");
}
Tensor *tren_le(Tensor *a, Tensor *b, bool in_place) {
    return binary_op(a, b, in_place, BIN_LE, backward_zeros, "BIN_LE");
}
Tensor *tren_eq(Tensor *a, Tensor *b, bool in_place) {
    return binary_op(a, b, in_place, BIN_EQ, backward_zeros, "BIN_EQ");
}
Tensor *tren_ne(Tensor *a, Tensor *b, bool in_place) {
    return binary_op(a, b, in_place, BIN_NE, backward_zeros, "BIN_NE");
}
Tensor *tren_gt(Tensor *a, Tensor *b, bool in_place) {
    return binary_op(a, b, in_place, BIN_GT, backward_zeros, "BIN_GE");
}
Tensor *tren_ge(Tensor *a, Tensor *b, bool in_place) {
    return binary_op(a, b, in_place, BIN_GE, backward_zeros, "BIN_GT");
}

// ---------------------------UNARY OPS----------------------------------------

static void (*un_fn(int_t op_id, BACKEND *backend))(Tensor *src, Tensor *dst) {
    switch (op_id) {
    case UN_NEG : return backend->neg_kernel;
    case UN_SIN : return backend->sin_kernel;
    case UN_COS : return backend->cos_kernel;
    case UN_TG  : return backend->tg_kernel;
    case UN_TGH : return backend->tgh_kernel;
    case UN_EXP : return backend->exp_kernel;
    case UN_LN  : return backend->ln_kernel;
    case UN_SQR : return backend->sqr_kernel;
    case UN_SQRT: return backend->sqrt_kernel;
    case UN_INV : return backend->inv_kernel;
    case UN_RELU: return backend->relu_kernel;
    default     : return NULL;
    }
}

static Tensor *unary_op(Tensor *a, bool in_place, int_t op_id, back_fun *backward, char *name) {
    TREN_CHECK(!a->requires_grad || !in_place, "In-place %s with grad required not allowed", name);
    DEVICE device = a->storage->device;
    Tensor *b = in_place ? a : tensor_create_empty(a->shape, a->ndims, device);
    b->requires_grad = a->requires_grad;
    BACKEND *backend = GET_BACKEND(device);
    un_fn(op_id, backend)(a, b);
    if (a->requires_grad) op_create(b, (Tensor *[]){a}, 1, backward, name, NULL, free);
    return b;
}

Tensor *tren_neg(Tensor *a, bool in_place) {
    return unary_op(a, in_place, UN_NEG, backward_neg, "Neg");
}
Tensor *tren_sin(Tensor *a, bool in_place) {
    return unary_op(a, in_place, UN_SIN, backward_sin, "Sin");
}
Tensor *tren_cos(Tensor *a, bool in_place) {
    return unary_op(a, in_place, UN_COS, backward_cos, "Cos");
}
Tensor *tren_tg(Tensor *a, bool in_place) {
    return unary_op(a, in_place, UN_TG, backward_tg, "Tg");
}
Tensor *tren_tgh(Tensor *a, bool in_place) {
    return unary_op(a, in_place, UN_TGH, backward_tgh, "Tgh");
}
Tensor *tren_sqr(Tensor *a, bool in_place) {
    return unary_op(a, in_place, UN_SQR, backward_sqr, "Sqr");
}
Tensor *tren_sqrt(Tensor *a, bool in_place) {
    return unary_op(a, in_place, UN_SQRT, backward_sqrt, "Sqrt");
}
Tensor *tren_inv(Tensor *a, bool in_place) {
    return unary_op(a, in_place, UN_INV, backward_inv, "Inv");
}
Tensor *tren_relu(Tensor *a, bool in_place) {
    return unary_op(a, in_place, UN_RELU, backward_relu, "RELU");
}
Tensor *tren_exp(Tensor *a, bool in_place) {
    return unary_op(a, in_place, UN_EXP, backward_exp, "Exp");
}
Tensor *tren_ln(Tensor *a, bool in_place) {
    return unary_op(a, in_place, UN_LN, backward_ln, "Ln");
}

// ---------------------------------------CONST OPS----------------------------------------------

static void (*con_fn(int_t op_id, BACKEND *backend))(Tensor *src, Tensor *dst, float c) {
    switch (op_id) {
    case CONST_ADD: return backend->add_c_kernel;
    case CONST_SUB: return backend->sub_c_kernel;
    case CONST_MUL: return backend->mul_c_kernel;
    case CONST_DIV: return backend->div_c_kernel;
    case CONST_LOG: return backend->log_c_kernel;
    case CONST_POW: return backend->pow_c_kernel;
    case CONST_LT : return backend->lt_c_kernel;
    case CONST_LE : return backend->le_c_kernel;
    case CONST_EQ : return backend->eq_c_kernel;
    case CONST_NE : return backend->ne_c_kernel;
    case CONST_GE : return backend->ge_c_kernel;
    case CONST_GT : return backend->gt_c_kernel;
    default       : return NULL;
    }
}

static Tensor *const_op(Tensor *a, const float c, bool in_place, int_t op_id, back_fun *backward,
                        char *name) {
    TREN_CHECK(!(a->requires_grad && in_place), "In-place %s with grad required not allowed", name);
    DEVICE device = a->storage->device;
    Tensor *b = in_place ? a : tensor_create_empty(a->shape, a->ndims, device);
    b->requires_grad = a->requires_grad;
    BACKEND *backend = GET_BACKEND(device);
    con_fn(op_id, backend)(a, b, c);
    if (a->requires_grad) {
        ConstCtx *ctx = (ConstCtx *)malloc(sizeof(ConstCtx));
        ctx->C = c;
        op_create(b, (Tensor *[]){a}, 1, backward, name, ctx, free);
    }
    return b;
}

Tensor *tren_add_const(Tensor *a, float c, bool in_place) {
    return const_op(a, c, in_place, CONST_ADD, backward_ones, "Add_const");
}
Tensor *tren_sub_const(Tensor *a, float c, bool in_place) {
    return const_op(a, c, in_place, CONST_SUB, backward_ones, "Sub_const");
}
Tensor *tren_mul_const(Tensor *a, float c, bool in_place) {
    return const_op(a, c, in_place, CONST_MUL, backward_mul_const, "Mul_const");
}
Tensor *tren_div_const(Tensor *a, float c, bool in_place) {
    return const_op(a, c, in_place, CONST_DIV, backward_div_const, "Div_const");
}
Tensor *tren_log_const(Tensor *a, float c, bool in_place) {
    return const_op(a, c, in_place, CONST_LOG, backward_log_const, "Log_const");
}
Tensor *tren_pow_const(Tensor *a, float c, bool in_place) {
    return const_op(a, c, in_place, CONST_POW, backward_pow_const, "Pow_const");
}
Tensor *tren_lt_const(Tensor *a, float c, bool in_place) {
    return const_op(a, c, in_place, CONST_LT, backward_zeros, "LT_const");
}
Tensor *tren_le_const(Tensor *a, float c, bool in_place) {
    return const_op(a, c, in_place, CONST_LE, backward_zeros, "LE_const");
}
Tensor *tren_eq_const(Tensor *a, float c, bool in_place) {
    return const_op(a, c, in_place, CONST_EQ, backward_zeros, "EQ_const");
}
Tensor *tren_ne_const(Tensor *a, float c, bool in_place) {
    return const_op(a, c, in_place, CONST_NE, backward_zeros, "NE_const");
}
Tensor *tren_ge_const(Tensor *a, float c, bool in_place) {
    return const_op(a, c, in_place, CONST_GE, backward_zeros, "GE_const");
}
Tensor *tren_gt_const(Tensor *a, float c, bool in_place) {
    return const_op(a, c, in_place, CONST_GT, backward_zeros, "GT_const");
}

// --------------------------REDUCE OPS------------------------------------

static void (*red_fn(int_t op_id, BACKEND *backend))(Tensor *src, Tensor *dst) {
    switch (op_id) {
    case RD_MIN: return backend->min_kernel;
    case RD_MAX: return backend->max_kernel;
    case RD_SUM: return backend->sum_kernel;
    default    : return NULL;
    }
}

static Tensor *reduce_op(Tensor *a, int_t *axes, int_t n_axes, float init_value, int_t op_id,
                         back_fun *backward, char *name) {
    TREN_CHECK(check_reduce(a->shape, a->ndims, axes, n_axes),
               "Invalid axes for reduce: tensor shape %s, axes %s", tuple_str(a->shape, a->ndims),
               tuple_str(axes, n_axes));
    DEVICE d = a->storage->device;
    BACKEND *backend = GET_BACKEND(d);
    int_t shape[MAX_DIM];
    calc_reduce_shape_sq(a, shape, axes, n_axes);
    Tensor *b = tensor_create_empty(shape, a->ndims - n_axes, d), b_temp;
    backend->fill_value_kernel(b, init_value);
    b->requires_grad = a->requires_grad;
    calc_reduce_shape_unsq(a, b_temp.shape, axes, n_axes);
    calc_strides(b_temp.shape, b_temp.strides, a->ndims);
    set_tensor_properties(&b_temp, NULL, NULL, a->ndims, b->numel, b->offset, b->storage);
    b_temp = make_broadcast_view(&b_temp, a->shape, a->ndims);
    Tensor *mask = b->requires_grad && (op_id == RD_MAX || op_id == RD_MIN)
                       ? tensor_create_empty(a->shape, a->ndims, d)
                       : NULL;
    red_fn(op_id, backend)(a, &b_temp);
    if (mask) {
        backend->eq_kernel(&b_temp, a, mask);
    }
    if (op_id == RD_MEAN) {
        int_t N = calc_numel_on_axes(a->shape, a->ndims, axes, n_axes);
        backend->mul_c_kernel(b, b, 1.f / N);
    }
    if (b->requires_grad) {
        ReduceCtx *ctx = (ReduceCtx *)malloc(sizeof(ReduceCtx));
        ctx->mask = mask;
        memcpy(ctx->axes, axes, n_axes * sizeof(int_t));
        ctx->n_axes = n_axes;
        op_create(b, (Tensor *[]){a}, 1, backward, name, ctx, free_ReduceCtx);
    }
    return b;
}
Tensor *tren_min(Tensor *a, int_t *axes, int_t n_axes) {
    return reduce_op(a, axes, n_axes, FLT_MAX, RD_MIN, backward_minmax, "Min");
}
Tensor *tren_max(Tensor *a, int_t *axes, int_t n_axes) {
    return reduce_op(a, axes, n_axes, -FLT_MAX, RD_MAX, backward_minmax, "Max");
}
Tensor *tren_sum(Tensor *a, int_t *axes, int_t n_axes) {
    return reduce_op(a, axes, n_axes, 0.f, RD_SUM, backward_sum, "Sum");
}
Tensor *tren_mean(Tensor *a, int_t *axes, int_t n_axes) {
    return reduce_op(a, axes, n_axes, 0.f, RD_MEAN, backward_mean, "Mean");
}

//----------------------------FILL OPS---------------------------------------------

/// @brief Заполнение тензора некоторым значением
/// @param t Тензор
/// @param c значение
void tren_fill_value(Tensor *t, const float c) {
    TREN_CHECK(tensor_is_contigous(t), "Cannot fill non-contigous tensor");
    GET_BACKEND(t->storage->device)->fill_value_kernel(t, c);
}

void tren_fill_from_array(Tensor *t, const float *arr, int_t numel) {
    TREN_CHECK(t->numel == numel, "Mismatch in tensor's and array numels: %lld and %lld", t->numel,
               numel);
    TREN_CHECK(tensor_is_contigous(t), "We don't fill non-contigous tensor");
    GET_BACKEND(t->storage->device)->fill_from_array_kernel(t, arr);
}
void tren_fill_random_uniform(Tensor *a, float min, float max) {
    TREN_CHECK(tensor_is_contigous(a), "Cannot fill non-contiguous tensor");
    GET_BACKEND(a->storage->device)->fill_random_uniform(a, min, max);
}
void tren_fill_random_normal(Tensor *a, float mean, float std) {
    TREN_CHECK(tensor_is_contigous(a), "Cannot fill non-contiguous tensor");
    GET_BACKEND(a->storage->device)->fill_random_normal(a, mean, std);
}

// -----------------------------MATMUL---------------------------------------

Tensor *tren_matmul(Tensor *a, Tensor *b) {
    TREN_CHECK(a->ndims > 0 && b->ndims > 0, "Cannot matmul %s and %s due one of them is scalar",
               tuple_str(a->shape, a->ndims), tuple_str(b->shape, b->ndims));
    int_t shape[MAX_DIM], ndims;
    TREN_CHECK(calc_matmul_result_shape(a, b, shape, &ndims),
               "Cannot matmul tensors with shapes %s and %s", tuple_str(a->shape, a->ndims),
               tuple_str(b->shape, b->ndims));
    BACKEND *backend = GET_BACKEND(a->storage->device);
    Tensor *c = tensor_create_empty(shape, ndims, a->storage->device);
    backend->fill_value_kernel(c, 0.f);
    Tensor a_view = make_matmul_broadcast_view(a, shape, ndims, true);
    Tensor b_view = make_matmul_broadcast_view(b, shape, ndims, false);
    backend->matmul_kernel(&a_view, &b_view, c);
    matmul_squeeze_shape(a, b, c->shape, c->strides, &c->ndims);
    if (a->requires_grad || b->requires_grad)
        op_create(c, (Tensor *[]){a, b}, 2, backward_matmul, "Matmul", NULL, free);
    return c;
}

// Tensor *tren_norm(Tensor *a, int_t *axes, int_t n_axes, float eps) {}

// ------------------OTHER FUNCTIONS-----------------------------
Tensor *tren_dropout(Tensor *a, float p, bool in_place) {
    TREN_CHECK(0 < p && p < 1, "Invalid probability: %.2f", p);
    TREN_CHECK(!a->requires_grad || !in_place, "In-place dropout with tensor requires grad");
    DEVICE d = a->storage->device;
    BACKEND *backend = GET_BACKEND(d);
    Tensor *res = in_place ? a : tensor_create_empty(a->shape, a->ndims, d);
    res->requires_grad = a->requires_grad;
    uint64_t seed = backend->get_seed();
    backend->dropout(a, res, p);
    if (res->requires_grad) {
        DropoutCtx *ctx = calloc(1, sizeof(DropoutCtx));
        ctx->seed = seed, ctx->p = p;
        op_create(res, (Tensor *[]){a}, 1, backward_dropout, "Dropout", ctx, free);
    }
    return res;
}

Tensor *tren_softmax(Tensor *a, int_t *axes, int_t n_axes) {
    TREN_CHECK(check_reduce(a->shape, a->ndims, axes, n_axes),
               "Invalid axes for reduce: tensor shape %s, axes %s", tuple_str(a->shape, a->ndims),
               tuple_str(axes, n_axes));
    DEVICE d = a->storage->device;
    BACKEND *backend = GET_BACKEND(d);
    int_t shape[MAX_DIM];
    calc_reduce_shape_unsq(a, shape, axes, n_axes);
    Tensor *sum = tensor_create_empty(shape, a->ndims, d), sum_t;
    Tensor *mx = tensor_create_empty(shape, a->ndims, d), mx_t;
    backend->fill_value_kernel(sum, 0.f), backend->fill_value_kernel(mx, -FLT_MAX);
    Tensor *res = tensor_create_empty(a->shape, a->ndims, d);
    res->requires_grad = a->requires_grad;
    sum_t = make_broadcast_view(sum, a->shape, a->ndims);
    mx_t = make_broadcast_view(mx, a->shape, a->ndims);
    backend->softmax(a, res, &mx_t, &sum_t);
    tensor_free(sum), tensor_free(mx);
    if (res->requires_grad) {
        ReduceCtx *ctx = (ReduceCtx *)malloc(sizeof(ReduceCtx));
        memcpy(ctx->axes, axes, n_axes * sizeof(int_t));
        ctx->n_axes = n_axes;
        op_create(res, (Tensor *[]){a}, 1, backward_softmax, "Softmax", ctx, free_ReduceCtx);
    }
    return res;
}

Tensor *tren_MSE(Tensor *pred, Tensor *target) {
    TREN_CHECK(tuples_equal(pred->shape, target->shape, pred->ndims, target->ndims),
               "Shape missmatch in MSE: %s and %s", tuple_str(pred->shape, pred->ndims),
               tuple_str(target->shape, target->ndims));
    TREN_CHECK(pred->storage->device == target->storage->device, "Device mismatch");
    DEVICE d = pred->storage->device;
    Tensor *res = tensor_create_empty(NULL, 1, d);
    Tensor res_t = make_broadcast_view(res, pred->shape, pred->ndims);
}
