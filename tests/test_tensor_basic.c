#include "test_utils.h"

static bool test_tensor_creation() {
    int_t shape[] = {2, 3};
    Tensor *t = tren_create(shape, 2, CPU_DEVICE, false);

    TEST_ASSERT(t != NULL, "Tensor creation failed");
    TEST_ASSERT(t->ndims == 2, "Expected ndims=2, got %lld", t->ndims);
    TEST_ASSERT(t->shape[0] == 2 && t->shape[1] == 3, "Wrong shape");
    TEST_ASSERT(t->numel == 6, "Expected numel=6, got %lld", t->numel);

    tren_free(t);
    return true;
}

static bool test_tensor_fill_and_get() {
    int_t shape[] = {2, 3};
    Tensor *t = tren_create(shape, 2, CPU_DEVICE, false);
    float test_data[] = {1, 2, 3, 4, 5, 6};
    tren_fill_from_array(t, test_data, 6);

    for (int_t i = 0; i < 6; ++i) {
        TEST_ASSERT_FLOAT_EQ(t->storage->data[i], test_data[i], 1e-5);
    }

    tren_free(t);
    return true;
}

static bool test_tensor_copy() {
    Tensor *src = create_test_tensor_2x3();
    Tensor *dst = tren_copy(src, CPU_DEVICE);
    TEST_ASSERT_TENSOR_EQ(src, dst, 1e-5);
    tren_free(src);
    tren_free(dst);
    return true;
}

static bool test_tensor_view() {
    int_t shape[] = {2, 3};
    Tensor *src = tren_create(shape, 2, CPU_DEVICE, false);
    float test_data[] = {1, 2, 3, 4, 5, 6};
    tren_fill_from_array(src, test_data, 6);

    int_t new_shape[] = {3, 2};
    Tensor *view = tren_view(src, new_shape, 2);

    TEST_ASSERT(view->storage == src->storage, "View should share storage");
    TEST_ASSERT(view->shape[0] == 3 && view->shape[1] == 2, "Wrong view shape");

    tren_free(view);
    tren_free(src);
    return true;
}

static bool test_tensor_slice() {
    Tensor *src = create_test_tensor_3x3();
    Slice slices[] = {{0, 2, 1, true}, {1, 3, 1, true}};
    Tensor *slice = tren_slice(src, slices, 2);

    TEST_ASSERT(slice->shape[0] == 2 && slice->shape[1] == 2, "Wrong slice shape");
    TEST_ASSERT_FLOAT_EQ(slice->storage->data[slice->offset + 0], 2, 1e-5);
    TEST_ASSERT_FLOAT_EQ(slice->storage->data[slice->offset + 1], 3, 1e-5);

    tren_free(slice);
    tren_free(src);
    return true;
}

static bool test_tensor_transpose() {
    Tensor *src = create_test_tensor_2x3();
    int_t order[] = {1, 0};
    Tensor *transposed = tren_transpose(src, order, 2);
    Tensor *double_trans = tren_transpose(transposed, order, 2);

    Tensor *eq = tren_eq(double_trans, src, false);

    TEST_ASSERT(transposed->shape[0] == 3 && transposed->shape[1] == 2, "Wrong transpose shape");
    for (int_t i = 0; i < eq->numel; ++i)
        TEST_ASSERT_FLOAT_EQ(eq->storage->data[i], 1.f, 1e-5);

    tren_free(transposed);
    tren_free(src);
    tren_free(double_trans);
    // tren_free(eq);
    return true;
}

void test_tensor_basic_suite(int *passed, int *failed, int *total) {
    RUN_TEST(test_tensor_creation);
    RUN_TEST(test_tensor_fill_and_get);
    RUN_TEST(test_tensor_copy);
    RUN_TEST(test_tensor_view);
    RUN_TEST(test_tensor_slice);
    RUN_TEST(test_tensor_transpose);
}