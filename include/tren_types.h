#pragma once
#include <stdbool.h>
#include <stdint.h>
typedef enum DEVICE { CPU_DEVICE, GPU_DEVICE } DEVICE;
typedef int int_t;
typedef struct Slice {
    int_t start;
    int_t end;
    int_t step;
    bool is_slice;
} Slice;