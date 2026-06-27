#include "tensor.h"
#include "../src/utils/error.h"
#include "../src/utils/utils.h"
#include "autograd.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Tensor *tensor_create_empty(const int_t *shape, int_t ndims, DEVICE d) {
    int_t numel = calc_numel(shape, ndims);
    Storage *s = storage_alloc(numel, d);
    Tensor *t = tensor_new_for_storage(s);
    calc_strides(shape, t->strides, ndims);
    set_tensor_properties(t, shape, NULL, ndims, numel, 0, NULL);
    return t;
}

void tensor_free(Tensor *t) {
    if (!t) return;
    if (--t->ref_count > 0) return;
    tensor_free(t->grad);
    if (t->creator) op_free(t->creator);
    if (--t->storage->ref_count == 0) {
        storage_free(t->storage);
    }
    free(t);
}

Tensor *tensor_view(Tensor *src, const int_t *new_shape, int_t new_ndims, bool in_place) {
    Tensor *dst = in_place ? src : tensor_new_for_storage(src->storage);
    calc_strides(new_shape, dst->strides, new_ndims);
    set_tensor_properties(dst, new_shape, NULL, new_ndims, src->numel, src->offset, NULL);
    return dst;
}

void tensor_print(const Tensor *t) {
    if (!t) return;
    print_tensor_info(t);
    print_tensor_recursive(t, 0, t->offset);
    printf("\n");
}

Tensor *tensor_slice(const Tensor *src, Slice *slices, int_t ndims) {
    Tensor *dst = tensor_new_for_storage(src->storage);
    int_t new_ndims = src->ndims, shape[MAX_DIM] = {0}, strides[MAX_DIM] = {0},
          offset = src->offset;
    for (int_t i = 0; i < ndims; ++i)
        if (!slices[i].is_slice) --new_ndims;

    memcpy(shape, src->shape + (src->ndims - new_ndims), new_ndims * sizeof(int_t));
    memcpy(strides, src->strides + (src->ndims - new_ndims), new_ndims * sizeof(int_t));
    int_t axis = 0;
    for (int_t i = 0; i < ndims; ++i) {
        Slice *s = &slices[i];
        if (s->is_slice) {
            shape[axis] = (s->end - s->start) / s->step;
            strides[axis] = src->strides[i] * s->step;
            offset += s->start * src->strides[i];
            axis++;
        } else {
            offset += s->start * src->strides[i];
        }
    }
    set_tensor_properties(dst, shape, strides, new_ndims, calc_numel(shape, new_ndims), offset,
                          NULL);
    return dst;
}

Tensor *tensor_transpose(Tensor *src, const int_t *order, bool in_place) {
    Tensor *dst = in_place ? src : tensor_new_for_storage(src->storage);
    calc_transpose_shape_and_strides(src, dst, order, src->ndims);
    set_tensor_properties(dst, NULL, NULL, src->ndims, src->numel, src->offset, NULL);
    return dst;
}

Tensor *tensor_copy(const Tensor *t, DEVICE d) {
    if (tensor_is_contigous(t)) {
    }
    Tensor *res = tensor_create_empty(t->shape, t->ndims, d);

    int_t inds[MAX_DIM] = {0}, offset = t->offset;
    for (int_t i = 0; i < t->numel; ++i) {
        res->storage->data[i] = t->storage->data[offset];
        next_idx(t, inds, &offset);
    }
    return res;
}

Tensor *tensor_squeeze(Tensor *src, bool in_place) {
    Tensor *dst = in_place ? src : tensor_new_for_storage(src->storage);
    calc_squeeze_shape_and_strides(src, dst);
    set_tensor_properties(dst, NULL, NULL, SKIP, src->numel, src->offset, NULL);
    return dst;
}
// Tensor *tensor_squeeze(Tensor *src, int_t axis, bool in_place) {
//     Tensor *dst = in_place ? src : tensor_new_for_storage(src->storage);
//     calc_squeeze_shape_and_strides(src, dst, axis);
//     set_tensor_properties(dst, NULL, NULL, SKIP, src->numel, src->offset, NULL);
//     return dst;
// }
