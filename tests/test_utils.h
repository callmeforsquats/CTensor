#pragma once
#include "../src/core/autograd.h"
#include "../src/core/tensor.h"
#include "tren.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Тестовые макросы
#define TEST_ASSERT(cond, msg, ...)                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            fprintf(stderr, "\033[31mTEST FAILED\033[0m: " msg "\n", ##__VA_ARGS__);               \
            fprintf(stderr, "  at %s:%d\n", __FILE__, __LINE__);                                   \
            return false;                                                                          \
        }                                                                                          \
    } while (0)

#define TEST_ASSERT_FLOAT_EQ(a, b, eps)                                                            \
    TEST_ASSERT(fabs((double)(a) - (double)(b)) < (eps), "Expected %f, got %f", (float)(b),        \
                (float)(a))

#define TEST_ASSERT_TENSOR_EQ(t1, t2, eps)                                                         \
    do {                                                                                           \
        TEST_ASSERT((t1)->ndims == (t2)->ndims, "ndims differ: %lld vs %lld", (t1)->ndims,         \
                    (t2)->ndims);                                                                  \
        for (int_t _i = 0; _i < (t1)->ndims; ++_i) {                                               \
            TEST_ASSERT((t1)->shape[_i] == (t2)->shape[_i], "Shape[%lld] differs: %lld vs %lld",   \
                        _i, (t1)->shape[_i], (t2)->shape[_i]);                                     \
        }                                                                                          \
        for (int_t _i = 0; _i < (t1)->numel; ++_i) {                                               \
            float _a = (t1)->storage->data[(t1)->offset + _i];                                     \
            float _b = (t2)->storage->data[(t2)->offset + _i];                                     \
            TEST_ASSERT_FLOAT_EQ(_a, _b, eps);                                                     \
        }                                                                                          \
    } while (0)

#define RUN_TEST(test)                                                                             \
    do {                                                                                           \
        bool res = test();                                                                         \
        printf("  %-40s ... ", #test);                                                             \
        fflush(stdout);                                                                            \
        if (res) {                                                                                 \
            printf("\033[32m+ PASSED\033[0m\n");                                                   \
            (*passed)++;                                                                           \
        } else {                                                                                   \
            printf("\033[31m- FAILED\033[0m\n");                                                   \
            (*failed)++;                                                                           \
        }                                                                                          \
        (*total)++;                                                                                \
    } while (0)

// Вспомогательные функции для создания тестовых тензоров
Tensor *create_test_tensor_2x2();
Tensor *create_test_tensor_2x3();
Tensor *create_test_tensor_3x3();

// Декларации тестовых suites
void test_tensor_basic_suite(int *passed, int *failed, int *total);
void test_ops_unary_suite(int *passed, int *failed, int *total);
void test_ops_binary_suite(int *passed, int *failed, int *total);
void test_autograd_suite(int *passed, int *failed, int *total);
void test_memory_suite(int *passed, int *failed, int *total);
void test_manual();
void test_ml();