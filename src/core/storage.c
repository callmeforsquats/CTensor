#include "../src/utils/error.h"
#include "tensor.h"
#include <stdlib.h>

typedef struct {
    void *ptr;
    int_t size;
} MemBlock;

#define POOL_SIZE 1024
MemBlock Pool[POOL_SIZE];
int_t count = 0;

void *p_alloc(int_t size) {
    for (int_t i = 0; i < count; ++i) {
        if (Pool[i].size >= size) {
            void *ptr = Pool[i].ptr;
            Pool[i] = Pool[--count];
            return ptr;
        }
    }
    return malloc(size);
}

void p_free(void *ptr, int_t size) {
    if (count < POOL_SIZE) Pool[count++] = (MemBlock){ptr, size};
    else free(ptr);
}

Storage *storage_alloc(int_t s, DEVICE d) {
    Storage *storage = (Storage *)calloc(1, sizeof(Storage));
    storage->size = s;
    storage->device = d;
    if (d == CPU_DEVICE) {
        storage->data = (float *)p_alloc(sizeof(float) * s);
        TREN_CHECK(storage->data != NULL, "Couldn't allocate %lld bytes of data on %s", s * 4,
                   d == CPU_DEVICE ? "CPU" : "GPU");
    }
    return storage;
}

void storage_free(Storage *s) {
    if (!s) return;
    if (s->device == CPU_DEVICE) {
        p_free(s->data, s->size);
    }
    free(s);
}