#include <crts.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "args.h"

extern SLAVE_FUN(sw_slave_mm_AB)(swptex_mmPara_t);
extern SLAVE_FUN(sw_slave_mm_ATB)(swptex_mmPara_t);
extern SLAVE_FUN(sw_slave_mm_ABT)(swptex_mmPara_t);
extern SLAVE_FUN(sw_sum_axis0)(swptex_mmPara_t);
extern SLAVE_FUN(sw_scale)(swptex_mmPara_t);
extern SLAVE_FUN(sw_merge)(swptex_mmPara_t);
extern SLAVE_FUN(sw_split)(swptex_mmPara_t);
extern SLAVE_FUN(sw_transpose_and_merge)(swptex_mmPara_t);
extern SLAVE_FUN(sw_split_and_transpose)(swptex_mmPara_t);
extern SLAVE_FUN(sw_softmax)(swptex_mmPara_t);
extern SLAVE_FUN(sw_dsoftmax)(swptex_mmPara_t);
extern SLAVE_FUN(sw_slave_bmm_AB_v1)(swptex_mmPara_t);
extern SLAVE_FUN(sw_slave_bmm_ABT_v1)(swptex_mmPara_t);
extern SLAVE_FUN(sw_slave_bmm_ATB_v1)(swptex_mmPara_t);
extern SLAVE_FUN(NN_384_0)(swptex_mmPara_t);
extern SLAVE_FUN(NN_0_384)(swptex_mmPara_t);
extern SLAVE_FUN(NN_6_64)(swptex_mmPara_t);
extern SLAVE_FUN(NN_64_6)(swptex_mmPara_t);
extern SLAVE_FUN(NT_384_0)(swptex_mmPara_t);
extern SLAVE_FUN(NT_0_384)(swptex_mmPara_t);
extern SLAVE_FUN(NT_64_6)(swptex_mmPara_t);
extern SLAVE_FUN(NT_6_64)(swptex_mmPara_t);
extern SLAVE_FUN(TN_384_0)(swptex_mmPara_t);
extern SLAVE_FUN(TN_0_384)(swptex_mmPara_t);
extern SLAVE_FUN(TN_64_6)(swptex_mmPara_t);
extern SLAVE_FUN(TN_6_64)(swptex_mmPara_t);
extern SLAVE_FUN(TN_32_32_128)(swptex_mmPara_t);
extern SLAVE_FUN(TN_v1_JYQ)(swptex_mmPara_t);
extern SLAVE_FUN(NN_v1_JYQ)(swptex_mmPara_t);
extern SLAVE_FUN(NN_384_0_end_finnal)(swptex_mmPara_t);
extern SLAVE_FUN(TN_384_0_end_finnal)(swptex_mmPara_t);
extern SLAVE_FUN(NT_384_0_end_finnal)(swptex_mmPara_t);
extern SLAVE_FUN(swptex_add_dma)(swptex_mmPara_t);
extern SLAVE_FUN(swptex_addcmul_dma)(swptex_mmPara_t);
extern SLAVE_FUN(swptex_addcdiv_dma)(swptex_mmPara_t);
extern SLAVE_FUN(sw_relu_dma)(swptex_mmPara_t);
extern SLAVE_FUN(sw_relu_backward_dma)(swptex_mmPara_t);

extern void *para_cross; // param on cross seg

int swptex_relu(void *src_, void *dst_, int len) {
#if 1
    swptex_mmPara para;
    para.src = (float *)src_;
    para.dst = (float *)dst_;
    para.len = len;
    para_cross = &para;

    int ret = athread_init_cgs();
    athread_spawn_cgs(sw_relu_dma, &para);
    athread_join_cgs();
#else
    float *src = (float *)src_;
    float *dst = (float *)dst_;
    int i;
    for (i = 0; i < len; ++i) {
        dst[i] = src[i] > 0.0 ? src[i] : 0.0;
    }
#endif
}

int swptex_relu_backward(void *x_, void *src_, void *dst_, int len) {
#if 1
    swptex_mmPara para;
    para.x = (float *)x_;
    para.src = (float *)src_;
    para.dst = (float *)dst_;
    para.len = len;
    para_cross = &para;

    int ret = athread_init_cgs();
    athread_spawn_cgs(sw_relu_backward_dma, &para);
    athread_join_cgs();
#else
    float *x = (float *)x_;
    float *src = (float *)src_;
    float *dst = (float *)dst_;
    int i;
    for (i = 0; i < len; ++i) {
        dst[i] = x[i] < 0.0 ? 0.0 : src[i];
    }
#endif
}

int swptex_add(void *x_, void *y_, int len, float ALPHA) {
    swptex_mmPara para;
    para.x = (float *)x_;
    para.y = (float *)y_;
    para.len = len;
    para.ALPHA = ALPHA;
    para_cross = &para;

    int ret = athread_init_cgs();
    athread_spawn_cgs(swptex_add_dma, &para);
    athread_join_cgs();
}

int swptex_addcmul(void *x_, void *t1_, void *t2_, int len, float value) {

    swptex_mmPara para;
    para.x = (float *)x_;
    para.t1 = (float *)t1_;
    para.t2 = (float *)t2_;
    para.len = len;
    para.value = value;
    para_cross = &para;

    int ret = athread_init_cgs();
    athread_spawn_cgs(swptex_addcmul_dma, &para);

    athread_join_cgs();
}

int swptex_addcdiv(void *x_, void *t1_, void *t2_, int len, float value) {

    swptex_mmPara para;
    para.x = (float *)x_;
    para.t1 = (float *)t1_;
    para.t2 = (float *)t2_;
    para.len = len;
    para.value = value;
    para_cross = &para;

    int ret = athread_init_cgs();
    athread_spawn_cgs(swptex_addcdiv_dma, &para);

    athread_join_cgs();
}

int swptex_mul(void *x_, size_t len, float ALPHA) {
    float *x = (float *)x_;
#if USE_SW_ADAMW
    swptex_mmPara para;
    para.x = x;
    para.len = len;
    para.scaling = ALPHA;
    para_cross = &para; // cross seg variable to pass param
    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_scale, &para);
    athread_join_cgs();
#else
    int i;
    for (i = 0; i < len; ++i) {
        x[i] *= ALPHA;
    }
#endif
}

void get_xxc(int *Mc_, int *Nc_, int *Kc_, int M, int N, int K) {
    int Mc = 0, Nc = 0, Kc = 0;
    int MAXLDM = 4096;
    if (M % 32 == 0 && N % 32 == 0 && K % 128 == 0) {
        Mc = 32;
        Nc = 32;
        Kc = 128;
    }
    else if (M % 64 == 0 && N % 64 == 0 && K % 64 == 0 && M != 64 && N != 64) {
        Mc = 64;
        Nc = 64;
        Kc = 64;
    } else if (M % 32 == 0 && N % 32 == 0 && K % 32 == 0) {
        Mc = 32;
        Nc = 32;
        Kc = 32;
        while (K % Kc == 0 && Kc <= 128) {
            Kc = Kc << 1;
        }
        Kc = Kc >> 1;
        while (N % Nc == 0 && Nc <= 128 && Nc * Kc <= MAXLDM) {
            Nc = Nc << 1;
        }
        Nc = Nc >> 1;
        while (M % Mc == 0 && Mc <= 128 && Mc * Kc <= MAXLDM &&
               Mc * Nc <= MAXLDM) {
            Mc = Mc << 1;
        }
        Mc = Mc >> 1;
    } 
    // else if (M % 16 == 0 && N % 16 == 0 && K % 16 == 0) {
    //     Mc = 16;
    //     Nc = 16;
    //     Kc = 16;
    //     while (K % Kc == 0 && Kc <= 128) {
    //         Kc = Kc << 1;
    //     }
    //     Kc = Kc >> 1;
    //     while (N % Nc == 0 && Nc <= 128 && Nc * Kc <= MAXLDM) {
    //         Nc = Nc << 1;
    //     }
    //     Nc = Nc >> 1;
    //     while (M % Mc == 0 && Mc <= 128 && Mc * Kc <= MAXLDM &&
    //            Mc * Nc <= MAXLDM) {
    //         Mc = Mc << 1;
    //     }
    //     Mc = Mc >> 1;
    // } 
    else if (M % 8 == 0 && N % 8 == 0 && K % 8 == 0) {
        Mc = 8;
        Nc = 8;
        Kc = 8;
        while (K % Kc == 0 && Kc <= 128) {
            Kc = Kc << 1;
        }
        Kc = Kc >> 1;
        while (N % Nc == 0 && Nc <= 128 && Nc * Kc <= MAXLDM) {
            Nc = Nc << 1;
        }
        Nc = Nc >> 1;
        while (M % Mc == 0 && Mc <= 128 && Mc * Kc <= MAXLDM &&
               Mc * Nc <= MAXLDM) {
            Mc = Mc << 1;
        }
        Mc = Mc >> 1;
    }

    *Mc_ = Mc;
    *Nc_ = Nc;
    *Kc_ = Kc;
}

int swptex_mm(const void *A, const void *B, void *C, size_t M, size_t N,
              size_t K, int transposeA, int transposeB) {
    swptex_mmPara para;
    para.A = A;
    para.B = B;
    para.C = C;
    para.M = M;
    para.N = N;
    para.K = K;
    para_cross = &para; // cross seg variable to pass param

    int ret = athread_init_cgs();
    int numM, numN, Mc = 0, Nc = 0, Kc = 0;
    unsigned long st, ed;
    st = athread_time_cycle();
    get_xxc(&Mc, &Nc, &Kc, M, N, K);
    ed = athread_time_cycle();

    para.Mc = Mc;
    para.Nc = Nc;
    para.Kc = Kc;
    int Ksize = Mc;
    Ksize = Nc < Ksize ? Nc : Ksize;
    Ksize = Kc < Ksize ? Kc : Ksize;
    Ksize = Ksize == 16 ? 8 : 8;
    para.Ksize = Ksize;
#if USW_SW_MM
    if (!transposeA && transposeB) { // NT
        if (Mc != 0) {
            numM = M / Mc;
            numN = N / Nc;
            if (numM >= 384) {
                ret = athread_spawn_cgs(NT_384_0, &para);
            } else if (numN >= 384) {
                ret = athread_spawn_cgs(NT_0_384, &para);
            } else if (numM >= 64) {
                ret = athread_spawn_cgs(NT_64_6, &para);
            } else if (numN >= 64) {
                ret = athread_spawn_cgs(NT_6_64, &para);
            } else {
                ret = athread_spawn_cgs(NT_384_0, &para);
            }
        } else {
            para.Mc = 8;
            para.Nc = 8;
            para.Kc = 8;
            para.Ksize = 8;
            ret = athread_spawn_cgs(NT_384_0_end_finnal, &para);
        }

    } else if (transposeA && !transposeB) { // TN
        if (M % 8 == 0 && N % 8 == 0 && K % 8 == 0) {
            Mc = M / 8;
            Nc = N / 8;
            Kc = K / 8;

            while (Mc * Nc > 4096) {
                if (Mc % 2 == 0)
                    Mc /= 2;
                else if (Nc % 2 == 0)
                    Nc /= 2;
            }
            while (Mc * Kc > 4096) {
                if (Mc % 2 == 0)
                    Mc /= 2;
                else if (Kc % 2 == 0)
                    Kc /= 2;
            }
            while (Nc * Kc > 4096) {
                if (Kc % 2 == 0)
                    Kc /= 2;
                else if (Nc % 2 == 0)
                    Nc /= 2;
            }
            para.Mc = Mc;
            para.Nc = Nc;
            para.Kc = Kc;
            ret = athread_spawn_cgs(TN_v1_JYQ, &para);
        } else if (M % 32 == 0 && N % 32 == 0 && K % 128 == 0) {
            para.Mc = 32;
            para.Nc = 32;
            para.Kc = 128;
            para.Ksize = 32;
            ret = athread_spawn_cgs(TN_384_0, &para);
        } else {
            if (Mc != 0) {
                numM = M / Mc;
                numN = N / Nc;
                if (numM >= 384) {
                    ret = athread_spawn_cgs(TN_384_0, &para);
                } else if (numN >= 384) {
                    ret = athread_spawn_cgs(TN_0_384, &para);
                } else if (numM >= 64) {
                    ret = athread_spawn_cgs(TN_64_6, &para);
                } else if (numN >= 64) {
                    ret = athread_spawn_cgs(TN_6_64, &para);
                } else {
                    // para.Mc = 8;
                    // para.Nc = 8;
                    // para.Kc = 8;
                    // ret = athread_spawn_cgs(TN_v1_JYQ, &para);
                    ret = athread_spawn_cgs(TN_64_6, &para);
                }
            } else {
                para.Mc = 8;
                para.Nc = 8;
                para.Kc = 8;
                para.Ksize = 8;
                ret = athread_spawn_cgs(TN_384_0_end_finnal, &para);
            }
        }
    } else if (!transposeA && !transposeB) { // NN
        if (Mc != 0) {
            numM = M / Mc;
            numN = N / Nc;
            if (numM >= 384) {
                ret = athread_spawn_cgs(NN_384_0, &para);
            } else if (numN >= 384) {
                ret = athread_spawn_cgs(NN_0_384, &para);
            } else if (numM >= 64) {
                ret = athread_spawn_cgs(NN_64_6, &para);
            } else if (numN >= 64) {
                ret = athread_spawn_cgs(NN_6_64, &para);
            } else {
                // para.Mc = 8;
                // para.Nc = 8;
                // para.Kc = 8;
                // para.Ksize = 8;
                // ret = athread_spawn_cgs(NN_v1_JYQ, &para);
                ret = athread_spawn_cgs(NN_64_6, &para);
            }
        } else {
            para.Mc = 8;
            para.Nc = 8;
            para.Kc = 8;
            para.Ksize = 8;
            ret = athread_spawn_cgs(NN_384_0_end_finnal, &para);
        }

    } else {
        printf("not supported\n");
        return 0;
    }
#else
    if (!transposeA && transposeB) {
        ret = athread_spawn_cgs(sw_slave_mm_ABT, &para);
    } else if (transposeA && !transposeB) {
        ret = athread_spawn_cgs(sw_slave_mm_ATB, &para);
    } else if (!transposeA && !transposeB) {
        ret = athread_spawn_cgs(sw_slave_mm_AB, &para);
    } else {
        printf("not supported\n");
        return 0;
    }
#endif
    athread_join_cgs();
}

int swptex_bmm(const void *A, const void *B, void *C, size_t batch, size_t M,
               size_t N, size_t K, int transposeA, int transposeB) {
    // int bn;
    // for (bn = 0; bn < batch; ++bn) {
    //   swptex_mm((float *)A + bn * M * K, (float *)B + bn * N * K,
    //             (float *)C + bn * M * N, M, N, K, transposeA, transposeB);
    // }
#if USE_SW_BMM
    swptex_mmPara para;
    para.A = A;
    para.B = B;
    para.C = C;
    para.M = M;
    para.N = N;
    para.K = K;
    para.bn = batch;
    para_cross = &para; // cross seg variable to pass param
    int ret = athread_init_cgs();
    if (M > 64 || N > 64 || K > 64)
        printf("MASTER ERROR [swptex_bmm] M=%ld, N=%ld, K=%ld\n", M, N, K);
    else {
        if (!transposeA && transposeB) {
            ret = athread_spawn_cgs(sw_slave_bmm_ABT_v1, &para);
        } else if (transposeA && !transposeB) {
            ret = athread_spawn_cgs(sw_slave_bmm_ATB_v1, &para);
        } else if (!transposeA && !transposeB) {
            ret = athread_spawn_cgs(sw_slave_bmm_AB_v1, &para);
        } else {
            printf("not supported\n");
        }
        athread_join_cgs();
    }
#else
    int bn;
    swptex_mmPara para;
    para.M = M;
    para.N = N;
    para.K = K;
    int ret = athread_init_cgs();

    for (bn = 0; bn < batch; ++bn) {
        para.A = (float *)A + bn * M * K;
        para.B = (float *)B + bn * N * K;
        para.C = (float *)C + bn * M * N;
        para_cross = &para; // cross seg variable to pass param
        if (!transposeA && transposeB) {
            ret = athread_spawn_cgs(sw_slave_mm_ABT, &para);
        } else if (transposeA && !transposeB) {
            ret = athread_spawn_cgs(sw_slave_mm_ATB, &para);
        } else if (!transposeA && !transposeB) {
            ret = athread_spawn_cgs(sw_slave_mm_AB, &para);
        } else {
            printf("not supported\n");
        }
        athread_join_cgs();
    }
#endif
}

int swptex_softmax(void *x_, size_t M, size_t N) {
    // inplace
    float *x = (float *)x_;
#if USE_SW_OTHER
    swptex_mmPara para;
    para.x = x;
    para.M = M;
    para.N = N;
    para_cross = &para; // cross seg variable to pass param
    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_softmax, &para);
    athread_join_cgs();
#else
    int i, j;
    float tmp, sum;
    for (i = 0; i < M; ++i) {
        tmp = x[i * N];
        for (j = 1; j < N; ++j) {
            if (x[i * N + j] > tmp) {
                tmp = x[i * N + j];
            }
        }
        sum = 0.f;
        for (j = 0; j < N; ++j) {
            x[i * N + j] = exp(x[i * N + j] - tmp);
            sum += x[i * N + j];
        }
        for (j = 0; j < N; ++j) {
            x[i * N + j] /= sum;
        }
    }
#endif
}

int swptex_dsoftmax(void *dy_, const void *y_, size_t M, size_t N) {
    // inplace
    float *dy = (float *)dy_;
    float *y = (float *)y_;
#if USE_SW_OTHER
    swptex_mmPara para;
    para.dy = dy;
    para.y = y;
    para.M = M;
    para.N = N;
    para_cross = &para; // cross seg variable to pass param
    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_dsoftmax, &para);
    athread_join_cgs();
#else
    int i, j;
    float tmp;
    for (i = 0; i < M; ++i) {
        tmp = 0.f;
        for (j = 0; j < N; ++j) {
            tmp += dy[i * N + j] * y[i * N + j];
        }
        for (j = 0; j < N; ++j) {
            dy[i * N + j] = (dy[i * N + j] - tmp) * y[i * N + j];
        }
    }
#endif
}

int swptex_split_and_transpose(void *QKV_, void *Q_, void *K_, void *V_,
                               size_t B, size_t N, size_t S, size_t D) {
    float *QKV = (float *)QKV_;
    float *Q = (float *)Q_;
    float *K = (float *)K_;
    float *V = (float *)V_;
#if USE_SW_OTHER
    swptex_mmPara para;
    para.QKV_data = QKV;
    para.Q_data = Q;
    para.K_data = K;
    para.V_data = V;
    para.b = B;
    para.n = N;
    para.s = S;
    para.d = D;
    para_cross = &para; // cross seg variable to pass param
    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_split_and_transpose, &para);
    athread_join_cgs();
#else
    int b, n, s;
    for (b = 0; b < B; ++b) {
        for (n = 0; n < N; ++n) {
            for (s = 0; s < S; ++s) {
                memcpy(Q + b * N * S * D + n * S * D + s * D,
                       QKV + n * D + s * N * D * 3 + b * S * N * D * 3,
                       D * sizeof(float));
                memcpy(K + b * N * S * D + n * S * D + s * D,
                       QKV + N * D + n * D + s * N * D * 3 + b * S * N * D * 3,
                       D * sizeof(float));
                memcpy(V + b * N * S * D + n * S * D + s * D,
                       QKV + N * D * 2 + n * D + s * N * D * 3 +
                           b * S * N * D * 3,
                       D * sizeof(float));
            }
        }
    }
#endif
}

int swptex_transpose_and_merge(void *QKV_, void *Q_, void *K_, void *V_,
                               size_t B, size_t N, size_t S, size_t D) {
    float *QKV = (float *)QKV_;
    float *Q = (float *)Q_;
    float *K = (float *)K_;
    float *V = (float *)V_;
#if USE_SW_OTHER
    swptex_mmPara para;
    para.QKV_data = QKV;
    para.Q_data = Q;
    para.K_data = K;
    para.V_data = V;
    para.b = B;
    para.n = N;
    para.s = S;
    para.d = D;
    para_cross = &para; // cross seg variable to pass param
    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_transpose_and_merge, &para);
    athread_join_cgs();
#else
    int b, n, s;
    for (b = 0; b < B; ++b) {
        for (n = 0; n < N; ++n) {
            for (s = 0; s < S; ++s) {
                memcpy(QKV + n * D + s * N * D * 3 + b * S * N * D * 3,
                       Q + b * N * S * D + n * S * D + s * D,
                       D * sizeof(float));
                memcpy(QKV + N * D + n * D + s * N * D * 3 + b * S * N * D * 3,
                       K + b * N * S * D + n * S * D + s * D,
                       D * sizeof(float));
                memcpy(
                    QKV + N * D * 2 + n * D + s * N * D * 3 + b * S * N * D * 3,
                    V + b * N * S * D + n * S * D + s * D, D * sizeof(float));
            }
        }
    }
#endif
}

int swptex_split(const void *QKV_, void *QKVT_, size_t B, size_t N, size_t S,
                 size_t D) {
    float *QKV = (float *)QKV_;
    float *QKVT = (float *)QKVT_;
#if USE_SW_OTHER
    swptex_mmPara para;
    para.b = B;
    para.n = N;
    para.s = S;
    para.d = D;
    para.QKV_data = QKV;
    para.QKVT_data = QKVT;

    para_cross = &para; // cross seg variable to pass param
    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_split, &para);
    athread_join_cgs();
#else
    int b, n, s;
    for (b = 0; b < B; ++b) {
        for (n = 0; n < N; ++n) {
            for (s = 0; s < S; ++s) {
                memcpy(QKVT + b * N * S * D + n * S * D + s * D,
                       QKV + n * D + s * N * D + b * S * N * D,
                       D * sizeof(float));
            }
        }
    }
#endif
}

int swptex_merge(const void *QKV_, void *QKVT_, size_t B, size_t N, size_t S,
                 size_t D) {
    float *QKV = (float *)QKV_;
    float *QKVT = (float *)QKVT_;
#if USE_SW_OTHER
    swptex_mmPara para;
    para.QKV_data = QKV;
    para.QKVT_data = QKVT;
    para.b = B;
    para.n = N;
    para.s = S;
    para.d = D;
    para_cross = &para; // cross seg variable to pass param
    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_merge, &para);
    athread_join_cgs();
#else
    int b, n, s;
    for (b = 0; b < B; ++b) {
        for (n = 0; n < N; ++n) {
            for (s = 0; s < S; ++s) {
                memcpy(QKVT + n * D + s * N * D + b * S * N * D,
                       QKV + b * N * S * D + n * S * D + s * D,
                       D * sizeof(float));
            }
        }
    }
#endif
}

int swptex_scale(void *x_, size_t len, float scaling) {
    float *x = (float *)x_;
#if USE_OTHER
    swptex_mmPara para;
    para.x = x;
    para.len = len;
    para.scaling = scaling;
    para_cross = &para; // cross seg variable to pass param
    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_scale, &para);
    athread_join_cgs();
#else
    int i;
    for (i = 0; i < len; ++i) {
        x[i] *= scaling;
    }
#endif
}

int sum_axis0(const void *src_, void *dst_, size_t M, size_t N) {
    float *src = (float *)src_;
    float *dst = (float *)dst_;
#if USE_SW_OTHER
    swptex_mmPara para;
    para.x = src;
    para.y = dst;
    para.M = M;
    para.N = N;
    para_cross = &para; // cross seg variable to pass param
    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_sum_axis0, &para);
    athread_join_cgs();
#else
    int i, j;
    for (j = 0; j < N; j++) {
        dst[j] = 0.0;
        for (i = 0; i < M; i++) {
            dst[j] += src[i * N + j];
        }
    }
#endif
}