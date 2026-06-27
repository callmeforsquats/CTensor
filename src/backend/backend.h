#pragma once
#include "tren_types.h"
#include <stdint.h>
#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#define RE __restrict
#define THREADY __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE inline __attribute__((always_inline))
#define RE restrict
#define THREADY _Thread_local
#else
#define FORCE_INLINE inline
#define RE restrict
#endif
#define MAX_OPERANDS 8

typedef struct Tensor Tensor;

typedef struct BACKEND {

    // ----------------BINARY KERNELS--------------------------
    void (*add_kernel)(Tensor *left, Tensor *right, Tensor *res);
    void (*sub_kernel)(Tensor *left, Tensor *right, Tensor *res);
    void (*mul_kernel)(Tensor *left, Tensor *right, Tensor *res);
    void (*div_kernel)(Tensor *left, Tensor *right, Tensor *res);
    void (*lt_kernel)(Tensor *left, Tensor *right, Tensor *res);
    void (*le_kernel)(Tensor *left, Tensor *right, Tensor *res);
    void (*eq_kernel)(Tensor *left, Tensor *right, Tensor *res);
    void (*ne_kernel)(Tensor *left, Tensor *right, Tensor *res);
    void (*ge_kernel)(Tensor *left, Tensor *right, Tensor *res);
    void (*gt_kernel)(Tensor *left, Tensor *right, Tensor *res);
    void (*assign_kernel)(Tensor *src, Tensor *dst);

    // ----------------UNARY KERNELS--------------------

    void (*neg_kernel)(Tensor *src, Tensor *dst);
    void (*sin_kernel)(Tensor *src, Tensor *dst);
    void (*cos_kernel)(Tensor *src, Tensor *dst);
    void (*tg_kernel)(Tensor *src, Tensor *dst);
    void (*tgh_kernel)(Tensor *src, Tensor *dst);
    void (*exp_kernel)(Tensor *src, Tensor *dst);
    void (*ln_kernel)(Tensor *src, Tensor *dst);
    void (*sqr_kernel)(Tensor *src, Tensor *dst);
    void (*sqrt_kernel)(Tensor *src, Tensor *dst);
    void (*inv_kernel)(Tensor *src, Tensor *dst);
    void (*relu_kernel)(Tensor *src, Tensor *dst);

    // --------------CONST KERNELS-----------------------------

    void (*add_c_kernel)(Tensor *src, Tensor *dst, float c);
    void (*sub_c_kernel)(Tensor *src, Tensor *dst, float c);
    void (*mul_c_kernel)(Tensor *src, Tensor *dst, float c);
    void (*div_c_kernel)(Tensor *src, Tensor *dst, float c);
    void (*lt_c_kernel)(Tensor *src, Tensor *dst, float c);
    void (*le_c_kernel)(Tensor *src, Tensor *dst, float c);
    void (*eq_c_kernel)(Tensor *src, Tensor *dst, float c);
    void (*ne_c_kernel)(Tensor *src, Tensor *dst, float c);
    void (*ge_c_kernel)(Tensor *src, Tensor *dst, float c);
    void (*gt_c_kernel)(Tensor *src, Tensor *dst, float c);
    void (*pow_c_kernel)(Tensor *src, Tensor *dst, float c);
    void (*log_c_kernel)(Tensor *src, Tensor *dst, float c);

    // -----------------REDUCE KERNELS------------------------------

    void (*sum_kernel)(Tensor *src, Tensor *dst);
    void (*min_kernel)(Tensor *src, Tensor *dst);
    void (*max_kernel)(Tensor *src, Tensor *dst);
    void (*prod_kernel)(Tensor *src, Tensor *dst);

    // ----------------FILL KERNELS--------------------

    void (*fill_value_kernel)(Tensor *A, const float c);
    void (*fill_from_array_kernel)(Tensor *A, const float *arr);
    void (*fill_range_kernel)(Tensor *A, int_t start, int_t step);
    void (*fill_random_uniform)(Tensor *A, float min, float max);
    void (*fill_random_normal)(Tensor *A, float mean, float std);

    // -------------OTHER KERNELS----------------------

    void (*dropout)(Tensor *src, Tensor *dst, float p);
    void (*softmax)(Tensor *src, Tensor *dst, Tensor *mx, Tensor *sum);
    void (*sum_prod_kernel)(Tensor **items, int_t n_items);

    // ---------------BACKWARD BINARY KERNELS---------------------------

    void (*back_add_kernel)(Tensor *in_grad, Tensor *out_grad);
    void (*back_neg_kernel)(Tensor *in_grad, Tensor *out_grad);
    void (*back_mul_kernel)(Tensor *in_grad, Tensor *out_grad, Tensor *other);
    void (*back_l_div_kernel)(Tensor *in_grad, Tensor *out_grad, Tensor *right);
    void (*back_r_div_kernel)(Tensor *in_grad, Tensor *out_grad, Tensor *right, Tensor *out);

    // ----------------BACKWARD UNARY KERNELS-------------------------------

    void (*back_sin_kernel)(Tensor *in_grad, Tensor *out_grad, Tensor *input);
    void (*back_cos_kernel)(Tensor *in_grad, Tensor *out_grad, Tensor *input);
    void (*back_tg_kernel)(Tensor *in_grad, Tensor *out_grad, Tensor *output);
    void (*back_tgh_kernel)(Tensor *in_grad, Tensor *out_grad, Tensor *output);
    void (*back_exp_kernel)(Tensor *in_grad, Tensor *out_grad, Tensor *output);
    void (*back_ln_kernel)(Tensor *in_grad, Tensor *out_grad, Tensor *input);
    void (*back_sqr_kernel)(Tensor *in_grad, Tensor *out_grad, Tensor *input);
    void (*back_sqrt_kernel)(Tensor *in_grad, Tensor *out_grad, Tensor *output);
    void (*back_relu_kernel)(Tensor *in_grad, Tensor *out_grad, Tensor *output);
    void (*back_inv_kernel)(Tensor *in_grad, Tensor *out_grad, Tensor *output);

    // ----------------------BACKWARD CONST KERNELS------------------------------

    void (*back_c_mul_kernel)(Tensor *in_grad, Tensor *out_grad, float c);
    void (*back_c_pow_kernel)(Tensor *in_grad, Tensor *out_grad, Tensor *input, float c);
    void (*back_c_log_kernel)(Tensor *in_grad, Tensor *out_grad, Tensor *input, float c);

    void (*back_dropout)(Tensor *in_grad, Tensor *out_grad, float p);
    void (*back_softmax)(Tensor *in_grad, Tensor *out_grad, Tensor *out, Tensor *sum_prod);

    void (*multi_reduce_kernel)(const Tensor *src, Tensor **dsts, uint32_t flags);

    void (*matmul_kernel)(Tensor *A, Tensor *B, Tensor *C);

    void (*set_seed)(uint64_t seed);
    uint64_t (*get_seed)();

    void (*adam_step_kernel)(Tensor *w, Tensor *m, Tensor *v, float lr, float beta1, float beta2,
                             float eps, float weight_decay, int_t step);

    void (*norm_kernel)(Tensor *x, Tensor *dst, Tensor *sum, Tensor *sum_sq, float eps, int_t N);

} BACKEND;

extern BACKEND CPU_NAIVE;
extern BACKEND CPU_OPTIM;
extern BACKEND GPU;

inline BACKEND *GET_BACKEND(DEVICE device) {
    switch (device) {
    case CPU_DEVICE: return &CPU_NAIVE;
    case GPU_DEVICE: return &CPU_NAIVE;
    default        : return &CPU_NAIVE;
    }
}
typedef enum { RD_SM = 1, RD_MEN, RD_SM_SQ, RD_SQ_MEAN, RD_MX, RD_MN, RD_PRD, RD_COUNT } REDUCTIONS;

typedef enum {

    RED_SUM = 1 << RD_SM,
    RED_MEAN = 1 << RD_MEN,
    RED_SUM_SQ = 1 << RD_SM_SQ,
    RED_SQ_MEAN = 1 << RD_SQ_MEAN,
    RED_MAX = 1 << RD_MX,
    RED_MIN = 1 << RD_MN,
    RED_PROD = 1 << RD_PRD,

} REDUCE_FLAGS;

typedef enum {

    BIN_ADD,
    BIN_SUB,
    BIN_MUL,
    BIN_DIV,
    BIN_LT,
    BIN_LE,
    BIN_EQ,
    BIN_NE,
    BIN_GE,
    BIN_GT,
    BIN_ASSIGN,

    UN_NEG,
    UN_SIN,
    UN_COS,
    UN_TG,
    UN_TGH,
    UN_EXP,
    UN_LN,
    UN_SQR,
    UN_SQRT,
    UN_INV,
    UN_RELU,

    CONST_ADD,
    CONST_SUB,
    CONST_MUL,
    CONST_DIV,
    CONST_LT,
    CONST_LE,
    CONST_EQ,
    CONST_NE,
    CONST_GE,
    CONST_GT,
    CONST_POW,
    CONST_LOG,

    RD_MIN,
    RD_MAX,
    RD_SUM,
    RD_PROD,
    RD_MEAN,
    RD_SUM_SQ,
    RD_MULTI,

    RAND_UN,
    RAND_NORM,

    DROPOUT,
    SOFTMAX,
    _SFTMX_1,
    _SFTMX_2,
    SUM_PROD,
    MSE,

    BW_ADD,
    BW_SUB,
    BW_MUL,
    BW_R_DIV,
    BW_L_DIV,

    BW_NEG,
    BW_SIN,
    BW_COS,
    BW_TG,
    BW_TGH,
    BW_EXP,
    BW_LN,
    BW_SQR,
    BW_SQRT,
    BW_INV,
    BW_RELU,

    BW_C_MUL,
    BW_C_POW,
    BW_C_LOG,

    BW_DROPOUT,
    BW_SOFTMAX,
    BW_MSE

} OpType;

typedef struct {
    union {
        struct {
            float scalar, threshold;
            int_t N;
        };
        struct {
            float alpha, beta;
        } leaky_relu;
        struct {
            float mean, std;
        } randn;
        struct {
            float min, max;
        } randu;
        struct {
            uint32_t *seed;
            float p, inv_p;
        } dropout;
        struct {
            float scalar;
            int_t mask;
        } red;
    };
} OpCtx;
