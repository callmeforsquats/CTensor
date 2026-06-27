#pragma once
#include "tren_types.h"
#include <stdbool.h>
#define MAX_DIM 16
#define SKIP -1

typedef struct OpNode OpNode;

typedef struct Storage {
    float *data;
    int_t size;
    DEVICE device;
    int_t ref_count;
} Storage;

typedef struct Tensor {
    Storage *storage;
    int_t shape[MAX_DIM];
    int_t strides[MAX_DIM];
    int_t ndims;
    int_t offset;
    int_t numel;
    bool requires_grad;
    int_t ref_count;
    bool grad_visited;
    struct Tensor *grad;
    OpNode *creator;
} Tensor;

Storage *storage_alloc(int_t size, DEVICE device);
void storage_free(Storage *s);

Tensor *tensor_create_empty(const int_t *shape, int_t ndims, DEVICE d);
Tensor *tensor_view(Tensor *src, const int_t *new_shape, int_t new_ndim, bool in_place);
void tensor_free(Tensor *t);
void op_free(OpNode *Node);
void tensor_print(const Tensor *t);
Tensor *tensor_slice(const Tensor *src, Slice *slices, int_t ndims);
Tensor *tensor_transpose(Tensor *src, const int_t *order, bool in_place);
Tensor *tensor_copy(const Tensor *t, DEVICE d);
Tensor *tensor_squeeze(Tensor *src, bool in_place);
// Tensor *tensor_squeeze(Tensor *src, int_t axis, bool in_place);
