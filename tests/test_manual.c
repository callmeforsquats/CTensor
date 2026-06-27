#include "test_utils.h"
void test_manual() {
    tren_seed_cpu_random(time(NULL));
    // Tensor *t = tren_create((int_t[]){10, 10}, 2, CPU_DEVICE, true);
    // tren_fill_random_normal(t, 0, 1.f);
    // tren_print(t);
    // Tensor *d = tren_dropout(t, 0.5, false);
    // tren_print(d);
    // backward(d);
    // tren_print(t->grad);

    Tensor *t = tren_create((int_t[]){3, 3, 3}, 3, CPU_DEVICE, true);
    tren_fill_random_normal(t, 0.f, 1.f);
    tren_print(t);
    int_t axes[] = {2, 1}, n_axes = 2;
    Tensor *sft = tren_softmax(t, axes, n_axes);
    tren_print(sft);
    Tensor *mx = tren_max(sft, axes, n_axes);
    tren_print(mx);
    backward(mx);
    tren_print(sft->grad), tren_print(t->grad);
    Tensor *t_plus = tren_add_const(t, 1e-1, false);
    Tensor *t_min = tren_add_const(t, -1e-1, false);
    Tensor *sft_plus = tren_softmax(t_plus, axes, n_axes);
    Tensor *sft_min = tren_softmax(t_min, axes, n_axes);
    tren_print(sft_plus), tren_print(sft_min);
    Tensor *num_grad = tren_mul_const(tren_sub(sft_plus, sft_min, false), 1.f / 2e-1, false);
    tren_print(num_grad);
}