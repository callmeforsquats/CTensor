#pragma once
#include "tren_types.h"
#include <stdbool.h>

typedef struct Tensor Tensor;
typedef struct Storage Storage;

#define SWAP(type, a, b)                                                                           \
    do {                                                                                           \
        type _tmp = (a);                                                                           \
        (a) = (b);                                                                                 \
        (b) = _tmp;                                                                                \
    } while (0)

void set_tensor_properties(Tensor *t, const int_t *shape, const int_t *strides, int_t ndims,
                           int_t numel, int_t offset, Storage *storage);
Tensor *tensor_new_for_storage(Storage *s);
bool tensor_is_contigous(const Tensor *t);
bool all_contiguous(Tensor **items, int_t n_items);
void calc_strides(const int_t *shape, int_t *strides, int_t ndims);
char *tuple_str(const int_t *tuple, int_t size);
char *slice_str(const Slice *s, const int_t size);
void print_tensor_recursive(const Tensor *t, const int_t dim, const int_t current_offset);
void print_tensor_info(const Tensor *t);
bool tuples_equal(const int_t *a, const int_t *b, const int_t s1, const int_t s2);
void calc_broadcast_result_shape(const Tensor *a, const Tensor *b, int_t *shape, int_t ndims);
int_t calc_numel(const int_t *shape, int_t ndims);
int_t calc_numel_on_axes(const int_t *shape, int_t ndims, const int_t *axes, int_t n_axes);
Tensor make_broadcast_view(const Tensor *t, int_t *shape, int_t ndims);
void calc_reduce_shape(const Tensor *t, int_t *shape, int_t *axes, int_t n_axes);
void calc_reduce_shape_sq(const Tensor *src, int_t *shape, int_t *axes, int_t n_axes);
void calc_reduce_shape_unsq(const Tensor *src, int_t *shape, int_t *axes, int_t n_axes);
bool check_reduce(int_t *shape, int_t ndims, int_t *axes, int_t n_axes);
void calc_transpose_shape_and_strides(const Tensor *src, Tensor *dst, const int_t *order,
                                      int_t axes);
void calc_squeeze_shape_and_strides(const Tensor *src, Tensor *dst);
bool check_slice(const Slice *slice, int_t shape);
bool check_shape(const int_t *shape, int_t ndims);
// Tensor *prepare_binary_op(Tensor *a, Tensor *b, Tensor *a_cast, Tensor *b_cast, bool in_place);
// Tensor *prepare_reduce_op(const Tensor *t, int_t *axes, int_t n_axes);
void next_idx(const Tensor *t, int_t *inds, int_t *cur_offset);
bool check_slices(Tensor *t, const Slice *slices, int_t shape);
void fill_range(int_t *arr, int_t limit);
bool calc_matmul_result_shape(Tensor *a, Tensor *b, int_t *shape, int_t *ndims);
void matmul_squeeze_shape(const Tensor *a, const Tensor *b, int_t *shape, int_t *strides,
                          int_t *ndims);
Tensor make_matmul_broadcast_view(Tensor *t, int_t *shape, int_t ndims, bool left);
Tensor *tensor_transpose_last2(Tensor *src, bool in_place);
int_t collapse_reduce_ndims(Tensor **items, int_t n_items, int_t *axes, int_t n_axes);
void collapse_dims(Tensor **items, int_t n_items, int_t stop);
bool check_last_stride(Tensor *items, int_t n_items);
find_reduce_axes(Tensor **items, int_t n_items, bool *axes, int_t *n_axes);