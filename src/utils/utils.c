#include "utils.h"
#include "../core/tensor.h"
#include "error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void set_tensor_properties(Tensor *t, const int_t *shape, const int_t *strides, int_t ndims,
                           int_t numel, int_t offset, Storage *storage) {
    if (ndims != SKIP) t->ndims = ndims;
    if (numel != SKIP) t->numel = numel;
    if (offset != SKIP) t->offset = offset;
    if (shape && ndims > 0) memcpy(t->shape, shape, ndims * sizeof(int_t));
    if (strides && ndims > 0) memcpy(t->strides, strides, ndims * sizeof(int_t));
    if (storage) t->storage = storage;
}

Tensor *tensor_new_for_storage(Storage *s) {
    Tensor *t = (Tensor *)calloc(1, sizeof(Tensor));
    t->storage = s;
    s->ref_count++;
    t->ref_count = 1;
    return t;
}

bool tensor_is_contigous(const Tensor *t) {
    int_t cur_stride = 1;
    for (int_t axis = t->ndims - 1; axis >= 0; --axis) {
        if (cur_stride != t->strides[axis]) return false;
        cur_stride *= t->shape[axis];
    }
    return true;
}

bool all_contiguous(Tensor **items, int_t n_items) {
    bool res = true;
    for (int_t i = 0; i < n_items; ++i) {
        res = tensor_is_contigous(items[i]);
        if (!res) return res;
    }
    return res;
}

void calc_strides(const int_t *shape, int_t *strides, int_t ndims) {
    int_t stride = 1;
    for (int_t i = ndims - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

char *tuple_str(const int_t *tuple, const int_t size) {
#define NUM_BUFFERS 4
#define BUF_SIZE 256
    static char buffers[NUM_BUFFERS][BUF_SIZE];
    static int buf_idx = 0;

    // Выбираем следующий буфер по кругу
    char *str = buffers[buf_idx];
    buf_idx = (buf_idx + 1) % NUM_BUFFERS;
    int pos = 0;
    pos += snprintf(str + pos, BUF_SIZE - pos, "(");
    for (int i = 0; i < size; ++i) {
        if (pos > BUF_SIZE) break;
        pos += snprintf(str + pos, BUF_SIZE - pos, "%lld%s", (long long)tuple[i],
                        (i < size - 1 ? "," : ")"));
    }
    return str;
}

char *slice_str(const Slice *s, const int_t size) {
    static char str[1024];
    const int_t buf_size = 1024;
    int pos = 0;
    pos += snprintf(str + pos, buf_size - pos, "(");
    for (int i = 0; i < size; ++i) {
        if (pos > buf_size) break;
        char *sep = i < size - 1 ? "," : "]";
        if (s[i].is_slice) {
            pos += snprintf(str + pos, buf_size - pos, "%lld:%lld:%lld%s", s[i].start, s[i].end,
                            s[i].step, sep);
        } else {
            pos += snprintf(str + pos, buf_size - pos, "%lld%s", s[i].start, sep);
        }
    }
    return str;
}

void print_tensor_recursive(const Tensor *t, const int_t dim, const int_t current_offset) {
    float *data = t->storage->data;
    if (t->ndims == 0) {
        printf("(%.2f)", data[current_offset]);
        return;
    }

    if (dim == t->ndims - 1) {
        printf("[");
        for (int_t i = 0; i < t->shape[dim]; i++) {
            int_t idx = current_offset + i * t->strides[dim];
            printf("%.2f%s", data[idx], (i == t->shape[dim] - 1 ? "" : ", "));
            if (i > 0 && i < t->shape[dim] - 3) {
                printf("... , ");
                i = t->shape[dim] - 3;
            }
        }
        printf("]");
    } else {
        printf("[");
        for (int_t i = 0; i < t->shape[dim]; i++) {
            print_tensor_recursive(t, dim + 1, current_offset + i * t->strides[dim]);
            if (i < t->shape[dim] - 1) {
                printf(",\n");
                for (int j = 0; j <= dim; j++)
                    printf(" ");
            }
            if (i > 0 && i < t->shape[dim] - 3) {
                printf("...\n");
                for (int j = 0; j <= dim; j++)
                    printf(" ");
                i = t->shape[dim] - 3;
            }
        }
        printf("]");
    }
}

void print_tensor_info(const Tensor *t) {
    if (t->storage->device != CPU_DEVICE) {
        printf("Tensor is on GPU. Copy to CPU first to print.\n");
        return;
    }
    if (t->ndims == 0) {
        printf("Tensor: scalar, device=CPU");
    } else {
        printf("Tensor: shape=(");
        for (int_t i = 0; i < t->ndims; ++i) {
            printf("%lld%s", t->shape[i], (i < t->ndims - 1) ? "," : "), ");
        }
        printf("strides=(");
        for (int_t i = 0; i < t->ndims; ++i) {
            printf("%lld%s", t->strides[i], (i < t->ndims - 1) ? "," : "), ");
        }
        printf("numel=%lld, device=%s, is_contigous=%s", t->numel,
               t->storage->device == CPU_DEVICE ? "CPU" : "GPU",
               tensor_is_contigous(t) ? "True" : "False");
    }
    printf("\n");
}

bool tuples_equal(const int_t *a, const int_t *b, int_t s1, int_t s2) {
    if (s1 != s2) return false;
    for (int i = 0; i < s1; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

void calc_broadcast_result_shape(const Tensor *a, const Tensor *b, int_t *shape, int_t ndims) {
    int_t a_i, b_i, a_sh, b_sh;
    for (int_t i = 0; i < ndims; ++i) {
        a_i = a->ndims - ndims + i, b_i = b->ndims - ndims + i;
        a_sh = a_i < 0 ? 1 : a->shape[a_i], b_sh = b_i < 0 ? 1 : b->shape[b_i];
        bool f = a_sh == 1 || b_sh == 1 || a_sh == b_sh;
        TREN_CHECK(f, "Cannot broadcast together tensors with shapes %s and %s",
                   tuple_str(a->shape, a->ndims), tuple_str(b->shape, b->ndims));
        shape[i] = max(a_sh, b_sh);
    }
}

int_t calc_numel(const int_t *shape, int_t ndims) {
    int_t numel = 1;
    for (int_t i = 0; i < ndims; ++i) {
        numel *= shape[i];
    }
    return numel;
}

int_t calc_numel_on_axes(const int_t *shape, int_t ndims, const int_t *axes, int_t n_axes) {
    int_t numel = 1;
    for (int_t i = 0; i < ndims; ++i)
        for (int_t j = 0; j < n_axes; ++j)
            if (axes[j] == i) numel *= shape[i];
    return numel;
}

Tensor make_broadcast_view(const Tensor *t, int_t *shape, int_t ndims) {
    Tensor view;
    int_t t_i;
    for (int_t i = 0; i < ndims; ++i) {
        t_i = t->ndims - ndims + i;
        view.shape[i] = shape[i];
        view.strides[i] = (t_i < 0 || t->shape[t_i] == 1 && shape[i] > 1) ? 0 : t->strides[t_i];
    }
    set_tensor_properties(&view, NULL, NULL, ndims, calc_numel(view.shape, ndims), t->offset,
                          t->storage);
    return view;
}

void calc_reduce_shape(const Tensor *src, int_t *shape, int_t *axes, int_t n_axes) {
    for (int_t i = 0; i < src->ndims; ++i) {
        shape[i] = src->shape[i];
        for (int_t j = 0; j < n_axes; ++j)
            if (i == axes[j]) shape[i] = 1;
    }
}

void calc_reduce_shape_sq(const Tensor *src, int_t *shape, int_t *axes, int_t n_axes) {
    int_t j = 0;
    for (int_t i = 0; i < src->ndims; ++i) {
        bool pres = false;
        for (int_t k = 0; k < n_axes; ++k)
            if (i == axes[k]) pres = true;
        if (!pres) shape[j++] = src->shape[i];
    }
}
void calc_reduce_shape_unsq(const Tensor *src, int_t *shape, int_t *axes, int_t n_axes) {
    for (int_t i = 0; i < src->ndims; ++i) {
        shape[i] = src->shape[i];
        for (int_t j = 0; j < n_axes; ++j)
            if (i == axes[j]) shape[i] = 1;
    }
}

void calc_transpose_shape_and_strides(const Tensor *src, Tensor *dst, const int_t *order,
                                      int_t axes) {
    bool pres[MAX_DIM] = {0};
    for (int_t i = 0; i < axes; ++i) {
        int_t idx = order[i];
        TREN_CHECK(!(idx < 0 || idx >= axes || pres[idx]),
                   "The axis %lld is out of bounds or duplicated in %lldD tensor", idx, axes);
        dst->shape[i] = src->shape[idx];
        dst->strides[i] = src->strides[idx];
        pres[idx] = true;
    }
}

void calc_squeeze_shape_and_strides(const Tensor *src, Tensor *dst) {
    int_t j = 0;
    for (int i = 0; i < src->ndims; ++i) {
        if (src->shape[i] > 1) {
            dst->shape[j] = src->shape[i];
            dst->strides[j] = src->strides[i];
            ++j;
        }
    }
    dst->ndims = j;
}

bool check_slice(const Slice *slice, int_t shape) {
    bool start = slice->start >= 0 && slice->start < shape;
    bool end = slice->end > slice->start || slice->end <= shape;
    bool range = slice->is_slice;
    bool step = slice->step > 0;
    return start && (!range || (end && step));
}

bool check_shape(const int_t *shape, int_t ndims) {
    for (int_t i = 0; i < ndims; ++i) {
        if (shape[i] <= 0) return false;
    }
    return true;
}

Tensor *prepare_binary_op(Tensor *a, Tensor *b, Tensor *a_cast, Tensor *b_cast, bool in_place) {
    TREN_CHECK(a->storage->device == b->storage->device, "Tensors are on different devices");
    int_t ndims = max(a->ndims, b->ndims), shape[MAX_DIM];
    calc_broadcast_result_shape(a, b, shape, ndims);
    TREN_CHECK(!in_place || tuples_equal(a->shape, shape, a->ndims, ndims),
               "Cannot make in-place operation: operand shape=%s, result shape=%s",
               tuple_str(a->shape, a->ndims), tuple_str(shape, ndims));
    *a_cast = make_broadcast_view(a, shape, ndims);
    *b_cast = make_broadcast_view(b, shape, ndims);
    Tensor *c = in_place ? a : tensor_create_empty(shape, ndims, a->storage->device);
    return c;
}

Tensor *prepare_reduce_op(const Tensor *t, int_t *axes, int_t n_axes) {
    TREN_CHECK(t->ndims >= n_axes, "Too many axes to reduce");
    for (int_t i = 0; i < n_axes; ++i)
        TREN_CHECK(axes[i] < t->ndims, "Cannot reduce %lld tensor on %lld axis", t->ndims, axes[i]);
    int_t shape[MAX_DIM];
    calc_reduce_shape(t, shape, axes, n_axes);
    return tensor_create_empty(shape, t->ndims, t->storage->device);
}

void next_idx(const Tensor *t, int_t *inds, int_t *cur_offset) {
    for (int_t dim = t->ndims - 1; dim >= 0; --dim) {
        if (++inds[dim] < t->shape[dim]) {
            *cur_offset += t->strides[dim];
            return;
        } else {
            inds[dim] = 0;
            *cur_offset -= (t->shape[dim] - 1) * t->strides[dim];
        }
    }
}

bool check_slices(Tensor *t, const Slice *slices, int_t n_slices) {
    if (t->ndims < n_slices) return false;
    for (int_t i = 0; i < n_slices; ++i) {
        bool cond = check_slice(&slices[i], t->shape[i]);
        if (!cond) return false;
    }
    return true;
}
void fill_range(int_t *arr, int_t limit) {
    for (int_t i = 0; i < limit; ++i)
        arr[i] = i;
}
bool check_reduce(int_t *shape, int_t ndims, int_t *axes, int_t n_axes) {
    if (ndims < n_axes) return false;
    for (int_t i = 0; i < n_axes - 1; ++i) {
        if (axes[i] < 0 || axes[i] >= ndims) return false;
        for (int_t j = i + 1; j < n_axes; ++j)
            if (axes[i] == axes[j]) return false;
    }
    return true;
}
bool calc_matmul_result_shape(Tensor *a, Tensor *b, int_t *shape, int_t *ndims) {
    Tensor *a_t = a->ndims > 1
                      ? a
                      : tensor_view(a, (int_t[]){1, (a->ndims == 1 ? a->shape[0] : 1)}, 2, false);
    Tensor *b_t = b->ndims > 1
                      ? b
                      : tensor_view(b, (int_t[]){(b->ndims == 1 ? b->shape[0] : 1), 1}, 2, false);

    int_t M = a_t->shape[a_t->ndims - 2], K1 = a_t->shape[a_t->ndims - 1],
          K2 = b_t->shape[b_t->ndims - 2], N = b_t->shape[b_t->ndims - 1];
    if (K1 != K2) return false;
    *ndims = max(a_t->ndims, b_t->ndims);
    int_t a_i, b_i, a_sh, b_sh;
    for (int_t i = 0; i < *ndims - 2; ++i) {
        a_i = a_t->ndims - *ndims + i, b_i = b_t->ndims - *ndims + i;
        a_sh = a_i < 0 ? 1 : a_t->shape[a_i], b_sh = b_i < 0 ? 1 : b_t->shape[b_i];
        bool f = a_sh == 1 || b_sh == 1 || a_sh == b_sh;
        if (!f) return false;
        shape[i] = max(a_sh, b_sh);
    }
    shape[*ndims - 2] = a_t->shape[a_t->ndims - 2];
    shape[*ndims - 1] = b_t->shape[b_t->ndims - 1];
    if (a->ndims < 2) tensor_free(a_t);
    if (b->ndims < 2) tensor_free(b_t);
    return true;
}
void matmul_squeeze_shape(const Tensor *a, const Tensor *b, int_t *strides, int_t *shape,
                          int_t *ndims) {
    if (a && a->ndims == 1) {
        shape[*ndims - 2] = shape[*ndims - 1];
        if (strides) strides[*ndims - 2] = strides[*ndims - 1];
        (*ndims)--;
    }
    if (b && b->ndims == 1) (*ndims)--;
}

Tensor make_matmul_broadcast_view(Tensor *t, int_t *shape, int_t ndims, bool left) {
    Tensor *tmp = t, view;
    if (t->ndims == 1)
        tmp = left ? tensor_view(t, (int_t[]){1, t->shape[0]}, 2, false)
                   : tensor_view(t, (int_t[]){t->shape[0], 1}, 2, false);
    int_t t_i = 0;
    for (int_t i = 0; i < ndims - 2; ++i) {
        t_i = t->ndims - ndims + i;
        view.shape[i] = shape[i];
        view.strides[i] = (t_i < 0 || t->shape[t_i] == 1 && shape[i] > 1) ? 0 : t->strides[t_i];
    }
    view.shape[ndims - 2] = tmp->shape[tmp->ndims - 2];
    view.strides[ndims - 2] = tmp->strides[tmp->ndims - 2];
    view.shape[ndims - 1] = tmp->shape[tmp->ndims - 1];
    view.strides[ndims - 1] = tmp->strides[tmp->ndims - 1];
    set_tensor_properties(&view, NULL, NULL, ndims, calc_numel(view.shape, ndims), tmp->offset,
                          tmp->storage);
    if (t->ndims == 1) tensor_free(tmp);
    return view;
}

Tensor *tensor_transpose_last2(Tensor *src, bool in_place) {
    Tensor *dst = in_place ? src : tensor_view(src, src->shape, src->ndims, false);
    SWAP(int_t, dst->shape[dst->ndims - 1], dst->shape[dst->ndims - 2]);
    SWAP(int_t, dst->strides[dst->ndims - 1], dst->strides[dst->ndims - 2]);
    return dst;
}

void collapse_dims(Tensor **items, int_t n_items, int_t stop) {
    int_t j = 0;
    for (int_t i = 1; i < stop; ++i) {
        bool can_collapse = true;
        for (int_t t = 0; t < n_items; ++t) {
            if (items[t]->strides[j] != items[t]->strides[i] * items[t]->shape[i]) {
                can_collapse = false;
                break;
            }
        }
        if (can_collapse) {
            for (int_t t = 0; t < n_items; ++t)
                items[t]->shape[j] *= items[t]->shape[i];
        } else {
            for (int_t t = 0; t < n_items; ++t) {
                j++;
                items[t]->shape[j] = items[t]->shape[i];
                items[t]->strides[j] = items[t]->strides[i];
            }
        }
    }
    for (int_t i = 0; i < n_items; ++i)
        items[i]->ndims = j + 1;
}

void collapse_reduce_ndims(Tensor **items, int_t n_items) {
    int j = 0;
    for (int_t i = 0; i < items[0]->ndims - 1; ++i) {
        bool can_collapse = true;
        for (int_t t = 0; t < n_items; ++t) {
            if (items[t]->strides[j] != items[t]->strides[i + 1] * items[t]->shape[i + 1]) {
                can_collapse = false;
                break;
            }
        }
        if (can_collapse) {
            for (int_t t = 0; t < n_items; ++t) {
                items[t]->shape[j] *= items[t]->shape[i];
                items[t]->strides[j] = items[t]->strides[i];
            }
        } else {
            for (int_t t = 0; t < n_items; ++t) {
                j++;
                items[t]->shape[j] = items[t]->shape[i];
                items[t]->strides[j] = items[t]->strides[i];
            }
        }
    }
    for (int_t t = 0; t < n_items; ++t)
        items[t]->ndims = j + 1;
}
bool check_last_stride(Tensor *items, int_t n_items) {
    for (int_t i = 0; i < n_items; ++i)
        if (items[i].strides[items[i].ndims - 1] > 1) return false;
    return true;
}
find_reduce_axes(Tensor **items, int_t n_items, bool *axes, int_t *n_axes) {
    int_t ndims = items[0]->ndims, *naxes = 0;
    for (int_t i = 1; i <= n_items; ++i)
        if (items[i]) {
            for (int_t j = ndims - 1; j >= 0; --j) {
                if (items[i]->strides[j] == 0) {
                    axes[j] = true;
                    (*naxes)++;
                }
            }
            break;
        }
}