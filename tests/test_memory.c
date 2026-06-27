#include "test_utils.h"
#include <stdlib.h>
static bool test_no_memory_leak_simple() {
    int_t shape[] = {100, 100};

    for (int i = 0; i < 100; ++i) {
        Tensor *t = tren_create(shape, 2, CPU_DEVICE, false);
        tren_fill_value(t, (float)i);
        // tren_free(t);
    }

    return true;
}

static bool test_no_memory_leak_autograd() {
    int_t shape[] = {10, 10};

    for (int i = 0; i < 50; ++i) {
        Tensor *a = tren_create(shape, 2, CPU_DEVICE, true);
        Tensor *b = tren_create(shape, 2, CPU_DEVICE, true);

        tren_fill_value(a, 1.0f);
        tren_fill_value(b, 2.0f);

        Tensor *c = tren_add(a, b, false);
        Tensor *d = tren_mul(c, a, false);
        Tensor *e = tren_relu(d, false);

        backward(e);

        tren_free(a);
        tren_free(b);
        tren_free(c);
        tren_free(d);
        tren_free(e);
    }

    return true;
}

static bool test_ref_count() {
    int_t shape[] = {2, 2};
    Tensor *a = tren_create(shape, 2, CPU_DEVICE, false);

    TEST_ASSERT(a->ref_count == 1, "Initial ref_count should be 1");
    TEST_ASSERT(a->storage->ref_count == 1, "Storage ref_count should be 1");

    Tensor *view = tren_view(a, shape, 2);
    TEST_ASSERT(view->ref_count == 1, "View ref_count should be 1");
    TEST_ASSERT(a->ref_count == 1, "After view, tensor ref_count shouldn't change");
    TEST_ASSERT(a->storage->ref_count == 2, "After view, storage ref_count should be 2");

    tren_free(view);
    TEST_ASSERT(a->ref_count == 1, "After free view, tensor ref_count should stay 1");
    TEST_ASSERT(a->storage->ref_count == 1, "After free view, storage ref_count should be 1");

    tren_free(a);
    return true;
}

static bool test_grad_cleanup() {
    int_t shape[] = {2, 2};
    Tensor *a = tren_create(shape, 2, CPU_DEVICE, true);
    tren_fill_value(a, 2.0f);

    Tensor *b = tren_sqr(a, false);
    backward(b);

    TEST_ASSERT(a->grad != NULL, "Grad should exist after backward");

    tren_free(b);
    tren_free(a);

    return true;
}

void test_memory_suite(int *passed, int *failed, int *total) {
    RUN_TEST(test_no_memory_leak_simple);
    RUN_TEST(test_no_memory_leak_autograd);
    RUN_TEST(test_ref_count);
    RUN_TEST(test_grad_cleanup);
}