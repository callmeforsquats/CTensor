#include "test_utils.h"
static bool test_add_grad() {
    int_t shape[] = {2, 2};
    Tensor *a = tren_create(shape, 2, CPU_DEVICE, true);
    Tensor *b = tren_create(shape, 2, CPU_DEVICE, true);

    tren_fill_value(a, 2.0f);
    tren_fill_value(b, 3.0f);

    Tensor *c = tren_add(a, b, false);
    backward(c);

    TEST_ASSERT(a->grad != NULL, "Grad for a is NULL");
    TEST_ASSERT(b->grad != NULL, "Grad for b is NULL");

    for (int_t i = 0; i < 4; ++i) {
        TEST_ASSERT_FLOAT_EQ(a->grad->storage->data[i], 1.0f, 1e-5);
        TEST_ASSERT_FLOAT_EQ(b->grad->storage->data[i], 1.0f, 1e-5);
    }

    tren_free(a);
    tren_free(b);
    tren_free(c);
    return true;
}

static bool test_mul_grad() {
    int_t shape[] = {2, 2};
    Tensor *a = tren_create(shape, 2, CPU_DEVICE, true);
    Tensor *b = tren_create(shape, 2, CPU_DEVICE, true);

    tren_fill_from_array(a, (float[]){1, 2, 3, 4}, 4);
    tren_fill_from_array(b, (float[]){2, 3, 4, 5}, 4);

    Tensor *c = tren_mul(a, b, false);
    backward(c);

    for (int_t i = 0; i < 4; ++i) {
        TEST_ASSERT_FLOAT_EQ(a->grad->storage->data[i], b->storage->data[i], 1e-5);
        TEST_ASSERT_FLOAT_EQ(b->grad->storage->data[i], a->storage->data[i], 1e-5);
    }

    tren_free(a);
    tren_free(b);
    tren_free(c);
    return true;
}

static bool test_relu_grad() {
    int_t shape[] = {2, 2};
    Tensor *a = tren_create(shape, 2, CPU_DEVICE, true);
    tren_fill_from_array(a, (float[]){-1, 2, -3, 4}, 4);

    Tensor *b = tren_relu(a, false);
    backward(b);

    float expected[] = {0, 1, 0, 1};
    for (int_t i = 0; i < 4; ++i) {
        TEST_ASSERT_FLOAT_EQ(a->grad->storage->data[i], expected[i], 1e-5);
    }

    tren_free(a);
    tren_free(b);
    return true;
}

static bool test_matmul_grad() {
    int_t shape_a[] = {2, 3};
    int_t shape_b[] = {3, 2};
    Tensor *a = tren_create(shape_a, 2, CPU_DEVICE, true);
    Tensor *b = tren_create(shape_b, 2, CPU_DEVICE, true);

    tren_fill_from_array(a, (float[]){1, 2, 3, 4, 5, 6}, 6);
    tren_fill_from_array(b, (float[]){1, 2, 3, 4, 5, 6}, 6);

    Tensor *c = tren_matmul(a, b);
    backward(c);

    TEST_ASSERT(a->grad != NULL, "Grad for a is NULL");
    TEST_ASSERT(b->grad != NULL, "Grad for b is NULL");

    // Проверяем, что градиенты не NaN
    for (int_t i = 0; i < a->numel; ++i) {
        TEST_ASSERT(!isnan(a->grad->storage->data[i]), "Gradient has NaN");
    }

    tren_free(a);
    tren_free(b);
    tren_free(c);
    return true;
}

static bool test_multiple_usage_grad() {
    // Тест: y = a * a (a используется дважды)
    int_t shape[] = {2, 2};
    Tensor *a = tren_create(shape, 2, CPU_DEVICE, true);
    tren_fill_from_array(a, (float[]){1, 2, 3, 4}, 4);

    Tensor *b = tren_mul(a, a, false);
    backward(b);

    // grad(a) = 2*a
    for (int_t i = 0; i < 4; ++i) {
        TEST_ASSERT_FLOAT_EQ(a->grad->storage->data[i], 2 * a->storage->data[i], 1e-5);
    }

    tren_free(a);
    tren_free(b);
    return true;
}

static bool test_chain_rule() {
    // Тест: y = (a + b) * c
    int_t shape[] = {2};
    Tensor *a = tren_create(shape, 1, CPU_DEVICE, true);
    Tensor *b = tren_create(shape, 1, CPU_DEVICE, true);
    Tensor *c = tren_create(shape, 1, CPU_DEVICE, true);

    tren_fill_from_array(a, (float[]){1, 2}, 2);
    tren_fill_from_array(b, (float[]){3, 4}, 2);
    tren_fill_from_array(c, (float[]){5, 6}, 2);

    Tensor *sum = tren_add(a, b, false);
    Tensor *y = tren_mul(sum, c, false);
    backward(y);

    for (int_t i = 0; i < 2; ++i) {
        TEST_ASSERT_FLOAT_EQ(a->grad->storage->data[i], c->storage->data[i], 1e-5);
        TEST_ASSERT_FLOAT_EQ(b->grad->storage->data[i], c->storage->data[i], 1e-5);
        TEST_ASSERT_FLOAT_EQ(c->grad->storage->data[i], a->storage->data[i] + b->storage->data[i],
                             1e-5);
    }

    tren_free(a);
    tren_free(b);
    tren_free(c);
    tren_free(sum);
    tren_free(y);
    return true;
}

void test_autograd_suite(int *passed, int *failed, int *total) {
    RUN_TEST(test_add_grad);
    RUN_TEST(test_mul_grad);
    RUN_TEST(test_relu_grad);
    RUN_TEST(test_matmul_grad);
    RUN_TEST(test_multiple_usage_grad);
    RUN_TEST(test_chain_rule);
}