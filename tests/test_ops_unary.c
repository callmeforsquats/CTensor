#include "test_utils.h"
static bool test_neg() {
    int_t shape[] = {2, 2};
    Tensor *a = tren_create(shape, 2, CPU_DEVICE, false);
    tren_fill_from_array(a, (float[]){1, -2, 3, -4}, 4);

    Tensor *b = tren_neg(a, false);

    float expected[] = {-1, 2, -3, 4};
    for (int_t i = 0; i < 4; ++i) {
        TEST_ASSERT_FLOAT_EQ(b->storage->data[i], expected[i], 1e-5);
    }

    tren_free(a);
    tren_free(b);
    return true;
}

static bool test_relu() {
    int_t shape[] = {2, 2};
    Tensor *a = tren_create(shape, 2, CPU_DEVICE, false);
    tren_fill_from_array(a, (float[]){-1, 2, -3, 4}, 4);

    Tensor *b = tren_relu(a, false);

    float expected[] = {0, 2, 0, 4};
    for (int_t i = 0; i < 4; ++i) {
        TEST_ASSERT_FLOAT_EQ(b->storage->data[i], expected[i], 1e-5);
    }

    tren_free(a);
    tren_free(b);
    return true;
}

static bool test_exp() {
    int_t shape[] = {2, 2};
    Tensor *a = tren_create(shape, 2, CPU_DEVICE, false);
    tren_fill_from_array(a, (float[]){0, 1, 2, 3}, 4);

    Tensor *b = tren_exp(a, false);

    for (int_t i = 0; i < 4; ++i) {
        TEST_ASSERT_FLOAT_EQ(b->storage->data[i], expf(a->storage->data[i]), 1e-5);
    }

    tren_free(a);
    tren_free(b);
    return true;
}

static bool test_sin_cos() {
    int_t shape[] = {2, 2};
    Tensor *a = tren_create(shape, 2, CPU_DEVICE, false);
    tren_fill_from_array(a, (float[]){0, 3.14159265f / 2, 3.14159265f, 3 * 3.14159265f / 2}, 4);

    Tensor *sin_a = tren_sin(a, false);
    Tensor *cos_a = tren_cos(a, false);

    TEST_ASSERT_FLOAT_EQ(sin_a->storage->data[0], 0, 1e-4);
    TEST_ASSERT_FLOAT_EQ(sin_a->storage->data[1], 1, 1e-4);
    TEST_ASSERT_FLOAT_EQ(cos_a->storage->data[0], 1, 1e-4);
    TEST_ASSERT_FLOAT_EQ(cos_a->storage->data[1], 0, 1e-4);

    tren_free(a);
    tren_free(sin_a);
    tren_free(cos_a);
    return true;
}

static bool test_reduce() {
    int_t shape[] = {2, 3, 4};
    Tensor *t = tren_range(0, 24, 1, CPU_DEVICE, false);
    Tensor *t2 = tren_view(t, shape, 3);
    Tensor *sum = tren_sum(t2, (int_t[]){0, 2}, 2);
    Tensor *max = tren_max(t2, (int_t[]){1, 2}, 2);
    return true;
    // tren_print(t2), tren_print(sum), tren_print(max);
}

void test_ops_unary_suite(int *passed, int *failed, int *total) {
    RUN_TEST(test_neg);
    RUN_TEST(test_relu);
    RUN_TEST(test_exp);
    RUN_TEST(test_sin_cos);
    RUN_TEST(test_reduce);
}