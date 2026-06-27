#include "test_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Конфигурация
#define TRAIN_SIZE 200
#define TEST_SIZE 50
#define HIDDEN_SIZE 64
#define EPOCHS 2000
#define LR 0.01f

// void train_sine() {
//     tren_seed_cpu_random(42);
//     // 1. Генерируем данные: x от 0 до 2*PI, y = sin(x)
//     float x_train[TRAIN_SIZE], y_train[TRAIN_SIZE];
//     float x_test[TEST_SIZE], y_test[TEST_SIZE];
//     for (int i = 0; i < TRAIN_SIZE; i++) {
//         x_train[i] = (float)i / TRAIN_SIZE * 2.0f * 3.14;
//         x_test[i] = (float)i / TEST_SIZE * 2.f * 3.14;
//         y_train[i] = sinf(x_train[i]);
//         y_test[i] = sinf(x_test[i]);
//     }
//     Tensor *X = tren_create((int_t[]){TRAIN_SIZE, 1}, 2, CPU_DEVICE, false);
//     Tensor *Y = tren_create((int_t[]){TRAIN_SIZE, 1}, 2, CPU_DEVICE, false);

//     tren_fill_from_array(X, x_train, TRAIN_SIZE);
//     tren_fill_from_array(X, x_train, TRAIN_SIZE);

//     // 2. Инициализация весов (желательно Xavier/He)
//     // Слой 1: 1 вход -> 64 скрытых
//     Tensor *W1 = tren_create((int_t[]){1, HIDDEN_SIZE}, 2, CPU_DEVICE, true);
//     Tensor *b1 = tren_create((int_t[]){1, HIDDEN_SIZE}, 2, CPU_DEVICE, true);
//     tren_fill_random_normal(W1, 0.f, 1.f);
//     tren_fill_value(b1, 0.f);

//     Tensor *W2 = tren_create((int_t[]){HIDDEN_SIZE, 1}, 2, CPU_DEVICE, true);
//     Tensor *b2 = tren_create((int_t[]){HIDDEN_SIZE, 1}, 2, CPU_DEVICE, true);
//     tren_fill_random_normal(W2, 0.f, 1.f);
//     tren_fill_value(b2, 0.f);

//     // 3. Цикл обучения
//     for (int epoch = 0; epoch < EPOCHS; epoch++) {
//         // --- Forward Pass ---
//         // layer1 = X @ W1 + b1
//         Tensor *mul1 = tren_matmul(X, W1);
//         Tensor *l1 = tren_add(mul1, b1, false);
//         // a1 = relu(l1)
//         Tensor *a1 = tren_relu(l1, false);
//         // out = a1 @ W2 + b2
//         Tensor *mul2 = tren_matmul(a1, W2);
//         Tensor *out = tren_add(mul2, b2, false);

//         // loss = MSE(out, Y)
//         // Tensor *loss = tren_mse_loss(out, Y);

//         // if (epoch % 200 == 0) {
//         //     printf("Epoch %d, Loss: %f\n", epoch, loss->data[0]);
//         // }

//         // // --- Backward Pass ---
//         // zero_grad((Tensor *[]){W1, b1, W2, b2}, 4);
//         // tren_backward(loss);

//         // --- SGD Step ---
//         // W = W - LR * grad
//         // В идеале сделать функцию optimizer_step(params, LR)
//         update_weights(W1, LR);
//         update_weights(b1, LR);
//         update_weights(W2, LR);
//         update_weights(b2, LR);

//         // Очистка графа (освобождение временных тензоров l1, a1, out, loss)
//         // В пет-проекте это обычно ручной вызов или простейший GC
//         graph_cleanup(loss);
//     }
// }

// // Вспомогательная функция для SGD
// void update_weights(Tensor *t, float lr) {
//     for (int i = 0; i < tensor_numel(t); i++) {
//         t->data[i] -= lr * t->grad->data[i];
//     }
// }

// void test_ml() {}