#include "test_utils.h"
static bool test_add() {
    Tensor *a = create_test_tensor_2x2();
    Tensor *b = create_test_tensor_2x2();
    tren_fill_value(b, 10.0f);

    Tensor *c = tren_add(a, b, false);

    float expected[] = {11, 12, 13, 14};
    for (int_t i = 0; i < 4; ++i) {
        TEST_ASSERT_FLOAT_EQ(c->storage->data[i], expected[i], 1e-5);
    }

    tren_free(a);
    tren_free(b);
    tren_free(c);
    return true;
}

static bool test_add_broadcast() {
    int_t shape_a[] = {2, 3};
    int_t shape_b[] = {3};
    Tensor *a = tren_create(shape_a, 2, CPU_DEVICE, false);
    Tensor *b = tren_create(shape_b, 1, CPU_DEVICE, false);

    tren_fill_from_array(a, (float[]){1, 2, 3, 4, 5, 6}, 6);
    tren_fill_from_array(b, (float[]){10, 20, 30}, 3);

    Tensor *c = tren_add(a, b, false);

    float expected[] = {11, 22, 33, 14, 25, 36};
    for (int_t i = 0; i < 6; ++i) {
        TEST_ASSERT_FLOAT_EQ(c->storage->data[i], expected[i], 1e-5);
    }

    tren_free(a);
    tren_free(b);
    tren_free(c);
    return true;
}

static bool test_mul() {
    int_t shape[] = {2, 2};
    Tensor *a = tren_create(shape, 2, CPU_DEVICE, false);
    Tensor *b = tren_create(shape, 2, CPU_DEVICE, false);

    tren_fill_from_array(a, (float[]){1, 2, 3, 4}, 4);
    tren_fill_from_array(b, (float[]){2, 3, 4, 5}, 4);

    Tensor *c = tren_mul(a, b, false);

    float expected[] = {2, 6, 12, 20};
    for (int_t i = 0; i < 4; ++i) {
        TEST_ASSERT_FLOAT_EQ(c->storage->data[i], expected[i], 1e-5);
    }

    tren_free(a);
    tren_free(b);
    tren_free(c);
    return true;
}

static bool test_matmul_2d() {
    int_t shape_a[] = {2, 3};
    int_t shape_b[] = {3, 2};
    Tensor *a = tren_create(shape_a, 2, CPU_DEVICE, false);
    Tensor *b = tren_create(shape_b, 2, CPU_DEVICE, false);

    tren_fill_from_array(a, (float[]){1, 2, 3, 4, 5, 6}, 6);
    tren_fill_from_array(b, (float[]){7, 8, 9, 10, 11, 12}, 6);

    Tensor *c = tren_matmul(a, b);

    TEST_ASSERT_FLOAT_EQ(c->storage->data[0], 58, 1e-5);
    TEST_ASSERT_FLOAT_EQ(c->storage->data[1], 64, 1e-5);
    TEST_ASSERT_FLOAT_EQ(c->storage->data[2], 139, 1e-5);
    TEST_ASSERT_FLOAT_EQ(c->storage->data[3], 154, 1e-5);

    tren_free(a);
    tren_free(b);
    tren_free(c);
    return true;
}
void test_ops_binary_suite(int *passed, int *failed, int *total) {
    RUN_TEST(test_add);
    RUN_TEST(test_add_broadcast);
    RUN_TEST(test_mul);
    RUN_TEST(test_matmul_2d);
}