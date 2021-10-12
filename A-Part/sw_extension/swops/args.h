#ifndef _ARGS_H
#define _ARGS_H

#define USE_KERNEL 1
#define USE_FAST_SIMD 0
#define BMM_KERNEL 1
#define SW_OPTIMIZE 1
#define USE_SW_OTHER 1
#define USE_SW_BMM 1
#define USW_SW_MM 1
#define USE_SW_ADAMW 1
#define SLAVE_LOG 0
#define USE_SW_RELU 0

typedef struct swptex_mmPara {
    float *A;
    float *B;
    float *C;
    int M;
    int N;
    int K;
    int bn;
    int Mc;
    int Nc;
    int Kc;
    int Ksize;

    float *t1;
    float *t2;
    float ALPHA;
    float value;

    float *QKV_data;
    float *QKVT_data;
    float *Q_data;
    float *K_data;
    float *V_data;
    float *src;
    float *dst;
    int b;
    int n;
    int s;
    int d;

    int len;
    float *x;
    float *dy;
    float *y;
    float scaling;
} swptex_mmPara, *swptex_mmPara_t;

#endif
