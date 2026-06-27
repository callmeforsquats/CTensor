#pragma once
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// Цвета для читаемости (опционально)
#define TREN_CLR_RED "\x1b[31m"
#define TREN_CLR_GRAY "\x1b[90m"
#define TREN_CLR_RESET "\x1b[0m"

// Общий макрос вывода (File, Line, Function + Message)
#define TREN_PRINT_ERROR(type, cond, fmt, ...)                                                     \
    do {                                                                                           \
        fprintf(stderr, TREN_CLR_RED "\n[TREN %s FAILED]" TREN_CLR_RESET "\n", type);              \
        fprintf(stderr, TREN_CLR_GRAY "  Location:  " TREN_CLR_RESET "%s:%d in %s()\n", __FILE__,  \
                __LINE__, __func__);                                                               \
        fprintf(stderr, TREN_CLR_GRAY "  Condition: " TREN_CLR_RESET "%s\n", #cond);               \
        fprintf(stderr, TREN_CLR_GRAY "  Message:   " TREN_CLR_RESET fmt "\n\n", ##__VA_ARGS__);   \
    } while (0)

// 1. TREN_CHECK — для API (шейпы, девайсы, указатели). Работает ВСЕГДА.
#define TREN_CHECK(cond, fmt, ...)                                                                 \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            TREN_PRINT_ERROR("CHECK", cond, fmt, ##__VA_ARGS__);                                   \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)
