#include "test_utils.h"

Tensor *create_test_tensor_2x2() {
    int_t shape[] = {2, 2};
    Tensor *t = tren_create(shape, 2, CPU_DEVICE, false);
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    tren_fill_from_array(t, data, 4);
    return t;
}

Tensor *create_test_tensor_2x3() {
    int_t shape[] = {2, 3};
    Tensor *t = tren_create(shape, 2, CPU_DEVICE, false);
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    tren_fill_from_array(t, data, 6);
    return t;
}

Tensor *create_test_tensor_3x3() {
    int_t shape[] = {3, 3};
    Tensor *t = tren_create(shape, 2, CPU_DEVICE, false);
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    tren_fill_from_array(t, data, 9);
    return t;
}