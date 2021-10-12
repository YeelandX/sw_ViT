#include "args.h"
#include "sgemm.h"
#include <crts.h>
#include <simd.h>
#include <slave.h>
#include <stdio.h>
#include <stdlib.h>

__cross void *para_cross = NULL;

#define thread_num (64 * 6)

#define __ALIGN512__ __attribute__((__aligned__(64)))
#define MEM64 64 * 64
#define MEM32 32 * 32
#define MEM8 8 * 8
#define MEM2048 2048
__thread_local float a_ldm[MEM64] __ALIGN512__, b_ldm[MEM64] __ALIGN512__,
    c_ldm[MEM64] __ALIGN512__;
__thread_local float aa_ldm[MEM64] __ALIGN512__, bb_ldm[MEM64] __ALIGN512__;
__thread_local float a_db_ldm[MEM64] __ALIGN512__, b_db_ldm[MEM64] __ALIGN512__;
__thread_local float at_ldm[MEM64] __ALIGN512__, bt_ldm[MEM64] __ALIGN512__;
__thread_local float a32_ldm[MEM32] __ALIGN512__, b32_ldm[MEM32] __ALIGN512__,
    c32_ldm[MEM32] __ALIGN512__;

__thread_local float x_ldm[MEM2048] __ALIGN512__, y_ldm[MEM2048] __ALIGN512__,
    dy_ldm[MEM2048] __ALIGN512__;
__thread_local float q_ldm[MEM2048] __ALIGN512__, k_ldm[MEM2048] __ALIGN512__,
    v_ldm[MEM2048] __ALIGN512__;
__thread_local float src_ldm[MEM2048] __ALIGN512__, dst_ldm[1];
__thread_local float a8_ldm[MEM8] __ALIGN512__, b8_ldm[MEM8] __ALIGN512__;
__thread_local double c8_ldm[MEM8] __ALIGN512__;
__thread_local volatile unsigned long reply, reply1;
__thread_local float *a_ptr[2], *b_ptr[2], *a_cur, *b_cur, *a_next, *b_next,
    *addr_a, *addr_b, *addr_c;
const int Mc = 64, Nc = 64, Kc = 64;

void swptex_add_dma(swptex_mmPara *_) {
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    float *x = para->x;
    float *y = para->y;
    float ALPHA = para->ALPHA;
    int len = para->len;

    int myid = CRTS_cgn * 64 + CRTS_tid;

    int slaveProcessNum = len / (thread_num);
    int segNum = 1024;
    int seg = slaveProcessNum / segNum;
    int segTotal = seg * segNum;
    int location = myid * slaveProcessNum;

    int i, j;
    for (i = 0; i < segTotal; i += segNum) {
        reply = 0;
        athread_dma_iget_stride(x_ldm, &x[location + i], segNum << 2, 0, 0,
                                &reply);
        athread_dma_iget_stride(y_ldm, &y[location + i], segNum << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 2);

        for (j = 0; j < segNum; j++) {
            x_ldm[j] += ALPHA * y_ldm[j];
        }

        reply = 0;
        athread_dma_iput_stride(&x[location + i], x_ldm, segNum << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);
    }

    if (i < slaveProcessNum) {
        int num = slaveProcessNum - i;

        reply = 0;
        athread_dma_iget_stride(x_ldm, &x[location + i], num << 2, 0, 0,
                                &reply);
        athread_dma_iget_stride(y_ldm, &y[location + i], num << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 2);

        for (j = 0; j < num; j++) {
            x_ldm[j] += ALPHA * y_ldm[j];
        }

        reply = 0;
        athread_dma_iput_stride(&x[location + i], x_ldm, num << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);
    }

    int tail = len % thread_num;

    if (myid == 0 && tail != 0) {
        reply = 0;
        athread_dma_iget_stride(x_ldm, &x[len - tail], tail << 2, 0, 0, &reply);
        athread_dma_iget_stride(y_ldm, &y[len - tail], tail << 2, 0, 0, &reply);
        athread_dma_wait_value(&reply, 2);

        for (j = 0; j < tail; j++) {
            x_ldm[j] += ALPHA * y_ldm[j];
        }

        reply = 0;
        athread_dma_iput_stride(&x[len - tail], x_ldm, tail << 2, 0, 0, &reply);
        athread_dma_wait_value(&reply, 1);
    }
}

void swptex_addcmul_dma(swptex_mmPara *_) {
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    float *x = para->x;
    float *t1 = para->t1;
    float *t2 = para->t2;

    float value = para->value;
    int len = para->len;

    int myid = CRTS_cgn * 64 + CRTS_tid;

    int slaveProcessNum = len / (thread_num);
    int segNum = 1024;
    int seg = slaveProcessNum / segNum;
    int segTotal = seg * segNum;
    int location = myid * slaveProcessNum;

    int i, j;
    for (i = 0; i < segTotal; i += segNum) {
        reply = 0;
        athread_dma_iget_stride(x_ldm, &x[location + i], segNum << 2, 0, 0,
                                &reply);
        athread_dma_iget_stride(k_ldm, &t1[location + i], segNum << 2, 0, 0,
                                &reply);
        athread_dma_iget_stride(v_ldm, &t2[location + i], segNum << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 3);
        for (j = 0; j < segNum; j++) {
            x_ldm[j] += value * k_ldm[j] * v_ldm[j];
        }

        reply = 0;
        athread_dma_iput_stride(&x[location + i], x_ldm, segNum << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);
    }

    if (i < slaveProcessNum) {
        int num = slaveProcessNum - i;

        reply = 0;
        athread_dma_iget_stride(x_ldm, &x[location + i], num << 2, 0, 0,
                                &reply);
        athread_dma_iget_stride(k_ldm, &t1[location + i], num << 2, 0, 0,
                                &reply);
        athread_dma_iget_stride(v_ldm, &t2[location + i], num << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 3);
        for (j = 0; j < num; j++) {
            x_ldm[j] += value * k_ldm[j] * v_ldm[j];
        }

        reply = 0;
        athread_dma_iput_stride(&x[location + i], x_ldm, num << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);
    }

    int tail = len % thread_num;

    if (myid == 0 && tail != 0) {
        reply = 0;
        athread_dma_iget_stride(x_ldm, &x[len - tail], tail << 2, 0, 0, &reply);
        athread_dma_iget_stride(k_ldm, &t1[len - tail], tail << 2, 0, 0,
                                &reply);
        athread_dma_iget_stride(v_ldm, &t2[len - tail], tail << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 3);

        for (j = 0; j < tail; j++) {
            x_ldm[j] += value * k_ldm[j] * v_ldm[j];
        }

        reply = 0;
        athread_dma_iput_stride(&x[len - tail], x_ldm, tail << 2, 0, 0, &reply);
        athread_dma_wait_value(&reply, 1);
    }
}

void swptex_addcdiv_dma(swptex_mmPara *_) {
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    float *x = para->x;
    float *t1 = para->t1;
    float *t2 = para->t2;

    float value = para->value;
    int len = para->len;

    int myid = CRTS_cgn * 64 + CRTS_tid;

    int slaveProcessNum = len / (thread_num);
    int segNum = 1024;
    int seg = slaveProcessNum / segNum;
    int segTotal = seg * segNum;
    int location = myid * slaveProcessNum;

    int i, j;
    for (i = 0; i < segTotal; i += segNum) {
        reply = 0;
        athread_dma_iget_stride(x_ldm, &x[location + i], segNum << 2, 0, 0,
                                &reply);
        athread_dma_iget_stride(k_ldm, &t1[location + i], segNum << 2, 0, 0,
                                &reply);
        athread_dma_iget_stride(v_ldm, &t2[location + i], segNum << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 3);
        for (j = 0; j < segNum; j++) {
            x_ldm[j] += value * k_ldm[j] / v_ldm[j];
        }

        reply = 0;
        athread_dma_iput_stride(&x[location + i], x_ldm, segNum << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);
    }

    if (i < slaveProcessNum) {
        int num = slaveProcessNum - i;

        reply = 0;
        athread_dma_iget_stride(x_ldm, &x[location + i], num << 2, 0, 0,
                                &reply);
        athread_dma_iget_stride(k_ldm, &t1[location + i], num << 2, 0, 0,
                                &reply);
        athread_dma_iget_stride(v_ldm, &t2[location + i], num << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 3);
        for (j = 0; j < num; j++) {
            x_ldm[j] += value * k_ldm[j] / v_ldm[j];
        }

        reply = 0;
        athread_dma_iput_stride(&x[location + i], x_ldm, num << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);
    }

    int tail = len % thread_num;

    if (myid == 0 && tail != 0) {
        reply = 0;
        athread_dma_iget_stride(x_ldm, &x[len - tail], tail << 2, 0, 0, &reply);
        athread_dma_iget_stride(k_ldm, &t1[len - tail], tail << 2, 0, 0,
                                &reply);
        athread_dma_iget_stride(v_ldm, &t2[len - tail], tail << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 3);

        for (j = 0; j < tail; j++) {
            x_ldm[j] += value * k_ldm[j] / v_ldm[j];
        }

        reply = 0;
        athread_dma_iput_stride(&x[len - tail], x_ldm, tail << 2, 0, 0, &reply);
        athread_dma_wait_value(&reply, 1);
    }
}

void sw_relu_dma(swptex_mmPara *_) {
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    float *src = para->src;
    float *dst = para->dst;
    int len = para->len;

    int myid = CRTS_cgn * 64 + CRTS_tid;

    int slaveProcessNum = len / (thread_num);
    int segNum = 1024;
    int seg = slaveProcessNum / segNum;
    int segTotal = seg * segNum;
    int location = myid * slaveProcessNum;

    int i, j;
    for (i = 0; i < segTotal; i += segNum) {
        reply = 0;
        athread_dma_iget_stride(x_ldm, &src[location + i], segNum << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);

        for (j = 0; j < segNum; j++) {
            y_ldm[j] = x_ldm[j] > 0.0 ? x_ldm[j] : 0.0;
        }

        reply = 0;
        athread_dma_iput_stride(&dst[location + i], y_ldm, segNum << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);
    }

    if (i < slaveProcessNum) {
        int num = slaveProcessNum - i;

        reply = 0;
        athread_dma_iget_stride(x_ldm, &src[location + i], num << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);

        for (j = 0; j < num; j++) {
            y_ldm[j] = x_ldm[j] > 0.0 ? x_ldm[j] : 0.0;
        }

        reply = 0;
        athread_dma_iput_stride(&dst[location + i], y_ldm, num << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);
    }

    int tail = len % thread_num;

    if (myid == 0 && tail != 0) {
        reply = 0;
        athread_dma_iget_stride(x_ldm, &src[len - tail], tail << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);

        for (j = 0; j < tail; j++) {
            y_ldm[j] = x_ldm[j] > 0.0 ? x_ldm[j] : 0.0;
        }

        reply = 0;
        athread_dma_iput_stride(&dst[len - tail], y_ldm, tail << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);
    }
}

int sw_relu_backward_dma(swptex_mmPara *_) {

    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    float *x = para->x;
    float *src = para->src;
    float *dst = para->dst;
    int len = para->len;

    int myid = CRTS_cgn * 64 + CRTS_tid;

    int slaveProcessNum = len / (thread_num);
    int segNum = 1024;
    int seg = slaveProcessNum / segNum;
    int segTotal = seg * segNum;
    int location = myid * slaveProcessNum;

    int i, j;
    for (i = 0; i < segTotal; i += segNum) {
        reply = 0;
        athread_dma_iget_stride(x_ldm, &x[location + i], segNum << 2, 0, 0,
                                &reply);
        athread_dma_iget_stride(v_ldm, &src[location + i], segNum << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 2);

        for (j = 0; j < segNum; j++) {
            y_ldm[j] = x_ldm[j] < 0.0 ? 0.0 : v_ldm[j];
        }

        reply = 0;
        athread_dma_iput_stride(&dst[location + i], y_ldm, segNum << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);
    }

    if (i < slaveProcessNum) {
        int num = slaveProcessNum - i;

        reply = 0;
        athread_dma_iget_stride(x_ldm, &x[location + i], num << 2, 0, 0,
                                &reply);
        athread_dma_iget_stride(v_ldm, &src[location + i], num << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 2);

        for (j = 0; j < num; j++) {
            y_ldm[j] = x_ldm[j] < 0.0 ? 0.0 : v_ldm[j];
        }

        reply = 0;
        athread_dma_iput_stride(&dst[location + i], y_ldm, num << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);
    }

    int tail = len % thread_num;

    if (myid == 0 && tail != 0) {

        reply = 0;
        athread_dma_iget_stride(x_ldm, &x[len - tail], tail << 2, 0, 0, &reply);
        athread_dma_iget_stride(v_ldm, &src[len - tail], tail << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 2);

        for (j = 0; j < tail; j++) {
            y_ldm[j] = x_ldm[j] < 0.0 ? 0.0 : v_ldm[j];
        }

        reply = 0;
        athread_dma_iput_stride(&dst[len - tail], y_ldm, tail << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);
    }
}

void padding_zero(float *src, float *dest, int M, int N, int newM, int newN) {
    int i, j;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            dest[i * newN + j] = src[i * N + j];
        }
        for (; j < newN; j++) {
            dest[i * newN + j] = 0;
        }
    }

    for (; i < newM; i++) {
        for (j = 0; j < newN; j++) {
            dest[i * newN + j] = 0;
        }
    }
}

void padding_restore(float *src, int M, int N, int oldM, int oldN) {
    int i, j;
    for (i = 0; i < oldM; i++) {
        for (j = 0; j < oldN; j++) {
            src[i * oldN + j] = src[i * N + j];
        }
    }
}

void padding_kernel(float *A, float *B, float *C, int M, int N, int K,
                    int Ksize) {
    if (Ksize == 8) {
        padding_8_kernel(A, B, C, M, N, K);
    } else if (Ksize == 32) {
        padding_32_kernel(A, B, C, M, N, K);
    } else if (Ksize == 64) {
#if USE_FAST_SIMD
        kernel_vlenmas_nn_64x64_asm_simd(A, B, C);
#elif USE_KERNEL
        kernel_vlenmas_nn_64x64_asm_fp0(A, B, C);
#endif
    }
}

void padding_32_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int i, j, k;
    for (i = 0; i < M; i += 32) {
        for (j = 0; j < N; j += 32) {
            for (k = 0; k < K; k += 32) {
                for (int ii = 0; ii < 32; ii++) {
                    for (int jj = 0; jj < 32; jj++) {
                        a32_ldm[ii * 32 + jj] = A[i * K + k + ii * K + jj];
                        b32_ldm[ii * 32 + jj] = B[k * N + j + ii * N + jj];
                        c32_ldm[ii * 32 + jj] = C[i * N + j + ii * N + jj];
                    }
                }
#if USE_FAST_SIMD
                kernel_vlenmas_nn_32x32_asm_simd(a32_ldm, b32_ldm,
                                                 c32_ldm); // fast_simd.c
#else
                kernel_vlenmas_nn_32x32_asm_fp0(a32_ldm, b32_ldm,
                                                c32_ldm); //汇编.c
#endif
                for (int ii = 0; ii < 32; ii++) {
                    for (int jj = 0; jj < 32; jj++) {
                        C[i * N + j + ii * N + jj] = c32_ldm[ii * 32 + jj];
                    }
                }
            }
        }
    }
}

void padding_8_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int i, j, k, ii, jj, kk;
    for (i = 0; i < M; i += 8) {
        for (j = 0; j < N; j += 8) {
            for (k = 0; k < K; k += 8) {
                for (ii = 0; ii < 8; ii++) {
                    for (jj = 0; jj < 8; jj++) {
                        a8_ldm[ii * 8 + jj] = A[i * K + k + ii * K + jj];
                        b8_ldm[ii * 8 + jj] = B[k * N + j + ii * N + jj];
                        c8_ldm[ii * 8 + jj] =
                            (double)C[i * N + j + ii * N + jj];
                    }
                }
#if USE_FAST_SIMD
                for (ii = 0; ii < 8; ii++) {
                    for (kk = 0; kk < 8; kk++) {
                        for (jj = 0; jj < 8; jj++) {
                            c8_ldm[ii * 8 + jj] +=
                                a8_ldm[ii * 8 + kk] * b8_ldm[kk * 8 + jj];
                        }
                    }
                }
#else
                sgemm_8_8_8(a8_ldm, b8_ldm, c8_ldm, 8, 8);
                asm volatile("memb\n");
#endif
                for (ii = 0; ii < 8; ii++) {
                    for (jj = 0; jj < 8; jj++) {
                        C[i * N + j + ii * N + jj] = c8_ldm[ii * 8 + jj];
                    }
                }
            }
        }
    }
}

void NN_384_0_end_finnal(swptex_mmPara *_) {
    unsigned long st, ed, allst, alled, caltime = 0, commtime = 0;
    allst = athread_stime_cycle();
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->K;
    int LDB = para->N;
    int LDC = para->N;
    int i, j, k, ii, jj, kk, i_end, j_end, k_end;
    int flag = 0;
    // 传入的分块大小
    int Mc = para->Mc, Nc = para->Nc, Kc = para->Kc;
    int Ksize = para->Ksize;

    floatv8 va, vb, vc;
    float A_Part;

    i_end = Mc;
    for (ii = myid * Mc; ii < M; ii += Mc * thread_num) {
        if (M - ii < Mc) {
            i_end = M - ii;
        }

        j_end = Nc;
        for (jj = 0; jj < N; jj += Nc) {
            if (N - jj < Nc) {
                j_end = N - jj;
            }

            for (kk = 0; kk < Mc * Nc; kk++)
                c_ldm[kk] = 0.0;

            k_end = Kc;
            for (kk = 0; kk < K; kk += Kc) {
                if (K - kk < Kc) {
                    k_end = K - kk;
                }
                reply = 0;
                st = athread_stime_cycle();

                athread_dma_iget_stride(a_ldm, &A[ii * LDA + kk],
                                        (i_end * k_end) << 2, k_end << 2,
                                        (LDA - k_end) << 2, &reply);
                athread_dma_iget_stride(b_ldm, &B[kk * LDB + jj],
                                        (k_end * j_end) << 2, j_end << 2,
                                        (LDB - j_end) << 2, &reply);
                athread_dma_wait_value(&reply, 2);
                ed = athread_stime_cycle();
                commtime += ed - st;

                st = athread_stime_cycle();
#if USE_KERNEL
                if (k_end < Kc || j_end < Nc || i_end < Mc) {
                    flag = 1;
                    padding_zero(a_ldm, aa_ldm, i_end, k_end, Mc, Kc);
                    padding_zero(b_ldm, bb_ldm, k_end, j_end, Kc, Nc);
                    padding_kernel(aa_ldm, bb_ldm, c_ldm, Mc, Nc, Kc, Ksize);
                } else {
                    padding_kernel(a_ldm, b_ldm, c_ldm, Mc, Nc, Kc, Ksize);
                }

#else
                for (i = 0; i < Mc; i++) {
                    for (k = 0; k < k_end; k++) {
                        A_Part = a_ldm[i * Kc + k];
                        va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part,
                                              A_Part, A_Part, A_Part, A_Part);
                        for (j = 0; j < Nc; j += 8) {
                            simd_load(vc, &c_ldm[i * Nc + j]);
                            simd_load(vb, &b_ldm[k * Nc + j]);
                            vc = simd_vmas(va, vb, vc);
                            simd_store(vc, &c_ldm[i * Nc + j]);
                        }
                    }
                }
#endif
                ed = athread_stime_cycle();
                caltime += ed - st;
            }
            if (flag) {
                flag = 0;
                padding_restore(c_ldm, Mc, Nc, i_end, j_end);
            }
#if USE_KERNEL && !USE_FAST_SIMD
            for (i = 0; i < Mc; i++) {
                for (j = 0; j < Nc; j++) {
                    c_ldm[i * Nc + j] *= (-1.0);
                }
            }
#endif

            st = athread_stime_cycle();
            reply = 0;
            athread_dma_iput_stride(&C[ii * LDC + jj], c_ldm,
                                    (i_end * j_end) << 2, j_end << 2,
                                    (LDC - j_end) << 2, &reply);
            athread_dma_wait_value(&reply, 1);
            ed = athread_stime_cycle();
            commtime += ed - st;
        }
    }
#if SLAVE_LOG
    alled = athread_stime_cycle();
    if (myid == 0)
        printf("##calculate time: %.3f, communication time: %.3f, all time: "
               "%.3f\n",
               1.0 * caltime / 2100000, 1.0 * commtime / 2100000,
               1.0 * (alled - allst) / 2100000);
#endif
}

void NT_384_0_end_finnal(swptex_mmPara *_) {
    unsigned long st, ed, allst, alled, caltime = 0, commtime = 0;
    allst = athread_stime_cycle();
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->K;
    int LDB = para->K;
    int LDC = para->N;
    int i, j, k, ii, jj, kk, i_end, j_end, k_end;
    int flag = 0;
    // 传入的分块大小
    int Mc = para->Mc, Nc = para->Nc, Kc = para->Kc;
    int Ksize = para->Ksize;

    floatv8 va, vb, vc;
    float A_Part;

    i_end = Mc;
    for (ii = myid * Mc; ii < M; ii += Mc * thread_num) {
        if (M - ii < Mc) {
            i_end = M - ii;
        }

        j_end = Nc;
        for (jj = 0; jj < N; jj += Nc) {
            if (N - jj < Nc) {
                j_end = N - jj;
            }

            for (kk = 0; kk < Mc * Nc; kk++)
                c_ldm[kk] = 0.0;

            k_end = Kc;
            for (kk = 0; kk < K; kk += Kc) {
                if (K - kk < Kc) {
                    k_end = K - kk;
                }
                reply = 0;
                st = athread_stime_cycle();

                athread_dma_iget_stride(a_ldm, &A[ii * LDA + kk],
                                        (i_end * k_end) << 2, k_end << 2,
                                        (LDA - k_end) << 2, &reply);
                athread_dma_iget_stride(b_ldm, &B[jj * LDB + kk],
                                        (k_end * j_end) << 2, k_end << 2,
                                        (LDB - k_end) << 2, &reply);
                athread_dma_wait_value(&reply, 2);
                ed = athread_stime_cycle();
                commtime += ed - st;

                st = athread_stime_cycle();
#if USE_KERNEL
                if (k_end < Kc || j_end < Nc || i_end < Mc) {
                    flag = 1;
                    padding_zero(a_ldm, aa_ldm, i_end, k_end, Mc, Kc);
                    padding_zero(b_ldm, bb_ldm, j_end, k_end, Nc, Kc);
                    for (k = 0; k < Kc; k++) {
                        for (j = 0; j < Nc; j++) {
                            bt_ldm[k * Nc + j] = bb_ldm[j * Kc + k];
                        }
                    }
                    padding_kernel(aa_ldm, bt_ldm, c_ldm, Mc, Nc, Kc, Ksize);
                } else {
                    for (k = 0; k < Kc; k++) {
                        for (j = 0; j < Nc; j++) {
                            bt_ldm[k * Nc + j] = b_ldm[j * Kc + k];
                        }
                    }
                    padding_kernel(a_ldm, bt_ldm, c_ldm, Mc, Nc, Kc, Ksize);
                }

#else
                for (i = 0; i < Mc; i++) {
                    for (k = 0; k < k_end; k++) {
                        A_Part = a_ldm[i * Kc + k];
                        va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part,
                                              A_Part, A_Part, A_Part, A_Part);
                        for (j = 0; j < Nc; j += 8) {
                            simd_load(vc, &c_ldm[i * Nc + j]);
                            simd_load(vb, &b_ldm[k * Nc + j]);
                            vc = simd_vmas(va, vb, vc);
                            simd_store(vc, &c_ldm[i * Nc + j]);
                        }
                    }
                }
#endif
                ed = athread_stime_cycle();
                caltime += ed - st;
            }
            if (flag) {
                flag = 0;
                padding_restore(c_ldm, Mc, Nc, i_end, j_end);
            }
#if USE_KERNEL && !USE_FAST_SIMD
            for (i = 0; i < Mc; i++) {
                for (j = 0; j < Nc; j++) {
                    c_ldm[i * Nc + j] *= (-1.0);
                }
            }
#endif

            st = athread_stime_cycle();
            reply = 0;
            athread_dma_iput_stride(&C[ii * LDC + jj], c_ldm,
                                    (i_end * j_end) << 2, j_end << 2,
                                    (LDC - j_end) << 2, &reply);
            athread_dma_wait_value(&reply, 1);
            ed = athread_stime_cycle();
            commtime += ed - st;
        }
    }
#if SLAVE_LOG
    alled = athread_stime_cycle();
    if (myid == 0)
        printf("##calculate time: %.3f, communication time: %.3f, all time: "
               "%.3f\n",
               1.0 * caltime / 2100000, 1.0 * commtime / 2100000,
               1.0 * (alled - allst) / 2100000);
#endif
}

void TN_384_0_end_finnal(swptex_mmPara *_) {
    unsigned long st, ed, allst, alled, caltime = 0, commtime = 0;
    allst = athread_stime_cycle();
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->M;
    int LDB = para->N;
    int LDC = para->N;
    int i, j, k, ii, jj, kk, i_end, j_end, k_end;
    int flag = 0;
    // 传入的分块大小
    int Mc = para->Mc, Nc = para->Nc, Kc = para->Kc;
    int Ksize = para->Ksize;

    floatv8 va, vb, vc;
    float A_Part;

    i_end = Mc;
    for (ii = myid * Mc; ii < M; ii += Mc * thread_num) {
        if (M - ii < Mc) {
            i_end = M - ii;
        }

        j_end = Nc;
        for (jj = 0; jj < N; jj += Nc) {
            if (N - jj < Nc) {
                j_end = N - jj;
            }

            for (kk = 0; kk < Mc * Nc; kk++)
                c_ldm[kk] = 0.0;

            k_end = Kc;
            for (kk = 0; kk < K; kk += Kc) {
                if (K - kk < Kc) {
                    k_end = K - kk;
                }
                reply = 0;
                st = athread_stime_cycle();

                athread_dma_iget_stride(a_ldm, &A[kk * LDA + ii],
                                        (i_end * k_end) << 2, i_end << 2,
                                        (LDA - i_end) << 2, &reply);
                athread_dma_iget_stride(b_ldm, &B[kk * LDB + jj],
                                        (k_end * j_end) << 2, j_end << 2,
                                        (LDB - j_end) << 2, &reply);
                athread_dma_wait_value(&reply, 2);
                ed = athread_stime_cycle();
                commtime += ed - st;

                st = athread_stime_cycle();
#if USE_KERNEL
                if (k_end < Kc || j_end < Nc || i_end < Mc) {
                    flag = 1;
                    padding_zero(a_ldm, aa_ldm, k_end, i_end, Kc, Mc);
                    padding_zero(b_ldm, bb_ldm, k_end, j_end, Kc, Nc);
                    for (i = 0; i < Mc; i++) {
                        for (k = 0; k < Kc; k++) {
                            at_ldm[k * Mc + i] = aa_ldm[i * Kc + k];
                        }
                    }
                    padding_kernel(at_ldm, bb_ldm, c_ldm, Mc, Nc, Kc, Ksize);
                } else {
                    for (i = 0; i < Mc; i++) {
                        for (k = 0; k < Kc; k++) {
                            at_ldm[k * Mc + i] = a_ldm[i * Kc + k];
                        }
                    }
                    padding_kernel(at_ldm, b_ldm, c_ldm, Mc, Nc, Kc, Ksize);
                }

#else
                for (i = 0; i < Mc; i++) {
                    for (k = 0; k < k_end; k++) {
                        A_Part = a_ldm[i * Kc + k];
                        va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part,
                                              A_Part, A_Part, A_Part, A_Part);
                        for (j = 0; j < Nc; j += 8) {
                            simd_load(vc, &c_ldm[i * Nc + j]);
                            simd_load(vb, &b_ldm[k * Nc + j]);
                            vc = simd_vmas(va, vb, vc);
                            simd_store(vc, &c_ldm[i * Nc + j]);
                        }
                    }
                }
#endif
                ed = athread_stime_cycle();
                caltime += ed - st;
            }
            if (flag) {
                flag = 0;
                padding_restore(c_ldm, Mc, Nc, i_end, j_end);
            }
#if USE_KERNEL && !USE_FAST_SIMD
            for (i = 0; i < Mc; i++) {
                for (j = 0; j < Nc; j++) {
                    c_ldm[i * Nc + j] *= (-1.0);
                }
            }
#endif

            st = athread_stime_cycle();
            reply = 0;
            athread_dma_iput_stride(&C[ii * LDC + jj], c_ldm,
                                    (i_end * j_end) << 2, j_end << 2,
                                    (LDC - j_end) << 2, &reply);
            athread_dma_wait_value(&reply, 1);
            ed = athread_stime_cycle();
            commtime += ed - st;
        }
    }
#if SLAVE_LOG
    alled = athread_stime_cycle();
    if (myid == 0)
        printf("##calculate time: %.3f, communication time: %.3f, all time: "
               "%.3f\n",
               1.0 * caltime / 2100000, 1.0 * commtime / 2100000,
               1.0 * (alled - allst) / 2100000);
#endif
}


void NN_v1_JYQ(swptex_mmPara *_) {
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;

    int Mc = para->Mc, Nc = para->Nc, Kc = para->Kc;

    int LDA = para->K;
    int LDB = para->N;
    int LDC = para->N;
    int i, j, k, ii, jj, kk, xi, i_k, i_j, k_j;
    floatv8 va, vb, vc;
    float A_Part;
    int STEP_A = _ROW * LDA * Mc + _COL * Kc;
    int STEP_B = _ROW * LDB * Kc + _COL * Nc;
    int STEP_C = _ROW * LDC * Mc + _COL * Nc;
    a_ptr[0] = a_ldm;
    a_ptr[1] = a_db_ldm;
    b_ptr[0] = b_ldm;
    b_ptr[1] = b_db_ldm;

    unsigned long st, ed, caltime = 0, commtime = 0;

    for (ii = _CGN * Mc * 8; ii < M; ii += Mc * 8 * 6) {
        for (jj = 0; jj < N; jj += Nc << 3) {
            addr_c = para->C + ii * N + jj;

            reply = 0;
            athread_dma_iget_stride(c_ldm, &addr_c[STEP_C], (Mc * Nc) << 2,
                                    Nc << 2, (LDC - Nc) << 2, &reply);
            athread_dma_wait_value(&reply, 1);

            kk = 0;
            addr_a = para->A + ii * K + kk;
            addr_b = para->B + kk * N + jj;

            a_cur = a_ldm;
            b_cur = b_ldm;

            reply1 = 0;
            athread_dma_iget_stride(a_cur, &addr_a[STEP_A], (Mc * Kc) << 2,
                                    Kc << 2, (LDA - Kc) << 2, &reply1);
            athread_dma_iget_stride(b_cur, &addr_b[STEP_B], (Kc * Nc) << 2,
                                    Nc << 2, (LDB - Nc) << 2, &reply1);
            int flag = 0;
            for (kk = Kc << 3; kk < K; kk += Kc << 3) {
                addr_a = para->A + ii * K + kk;
                addr_b = para->B + kk * N + jj;
                flag = !flag;
                a_next = a_ptr[flag & 0x1];
                b_next = b_ptr[flag & 0x1];
                athread_dma_wait_value(&reply1, 2);

                reply1 = 0;
                athread_dma_iget_stride(a_next, &addr_a[STEP_A], (Mc * Kc) << 2,
                                        Kc << 2, (LDA - Kc) << 2, &reply1);
                athread_dma_iget_stride(b_next, &addr_b[STEP_B], (Kc * Nc) << 2,
                                        Nc << 2, (LDB - Nc) << 2, &reply1);

                for (xi = 0; xi <= 7; xi++) {
                    reply = 0;
                    athread_ssync_array();

                    if (_COL == xi)
                        athread_rma_row_ibcast(aa_ldm, a_cur, (Mc * Kc) << 2, 0,
                                               &reply);
                    if (_ROW == xi)
                        athread_rma_col_ibcast(bb_ldm, b_cur, (Kc * Nc) << 2, 0,
                                               &reply);
                    athread_rma_wait_value(&reply, 2);
                    athread_ssync_array();

                    // cal kernel
                    for (i = 0; i < Mc; i++) {
                        for (k = 0; k < Kc; k++) {
                            for (j = 0; j < Nc; j++) {
                                c_ldm[i * Nc + j] +=
                                    aa_ldm[i * Kc + k] * bb_ldm[k * Nc + j];
                            }
                        }
                    }
                }
                a_cur = a_next;
                b_cur = b_next;
            }
            athread_dma_wait_value(&reply1, 2);
            for (int xi = 0; xi <= 7; xi++) {
                reply = 0;
                athread_ssync_array();
                if (_COL == xi)
                    athread_rma_row_ibcast(aa_ldm, a_cur, (Mc * Kc) << 2, 0,
                                           &reply);
                if (_ROW == xi)
                    athread_rma_col_ibcast(bb_ldm, b_cur, (Kc * Nc) << 2, 0,
                                           &reply);
                athread_rma_wait_value(&reply, 2);
                athread_ssync_array();
                for (i = 0; i < Mc; i++) {
                    for (k = 0; k < Kc; k++) {
                        for (j = 0; j < Nc; j++) {
                            c_ldm[i * Nc + j] +=
                                aa_ldm[i * Kc + k] * bb_ldm[k * Nc + j];
                        }
                    }
                }
            }
            athread_ssync_array();
            reply = 0;
            athread_dma_iput_stride(&addr_c[STEP_C], c_ldm, (Mc * Nc) << 2,
                                    Nc << 2, (LDC - Nc) << 2, &reply);
            athread_dma_wait_value(&reply, 1);
        }
    }
}

void TN_v1_JYQ(swptex_mmPara *_) {
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    int Mc = para->Mc;
    int Nc = para->Nc;
    int Kc = para->Kc;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->M;
    int LDB = para->N;
    int LDC = para->N;
    int i, j, k, ii, jj, kk, xi, i_k, i_j, k_j, p;
    floatv8 va, vb, vc;
    float A_Part;
    int STEP_A = _COL * LDA * Kc + _ROW * Mc;
    int STEP_B = _ROW * LDB * Kc + _COL * Nc;
    int STEP_C = _ROW * LDC * Mc + _COL * Nc;
    a_cur = a_ldm;
    b_cur = b_ldm;

    unsigned long st, ed, caltime = 0, commtime = 0;

    for (ii = _CGN * Mc * 8; ii < M; ii += Mc * 8 * 6) {
        for (jj = 0; jj < N; jj += Nc << 3) {
            st = athread_stime_cycle();
            addr_c = para->C + ii * LDC + jj;
            reply = 0;

            athread_dma_iget_stride(c_ldm, &addr_c[STEP_C], (Mc * Nc) << 2,
                                    Nc << 2, (LDC - Nc) << 2, &reply);
            athread_dma_wait_value(&reply, 1);

            ed = athread_stime_cycle();
            commtime += ed - st;

            for (kk = 0; kk < K; kk += Kc << 3) {
                addr_a = para->A + kk * LDA + ii;
                addr_b = para->B + kk * LDB + jj;
                st = athread_stime_cycle();
                reply1 = 0;
                athread_dma_iget_stride(a_cur, &addr_a[STEP_A], (Mc * Kc) << 2,
                                        Mc << 2, (LDA - Mc) << 2, &reply1);
                athread_dma_iget_stride(b_cur, &addr_b[STEP_B], (Kc * Nc) << 2,
                                        Nc << 2, (LDB - Nc) << 2, &reply1);
                athread_dma_wait_value(&reply1, 2);
                ed = athread_stime_cycle();
                commtime += ed - st;
                for (xi = 0; xi <= 7; xi++) {
                    st = athread_stime_cycle();
                    reply = 0;
                    //                    sync_64spe();
                    athread_ssync_array();
                    if (_COL == xi)
                        athread_rma_row_ibcast(aa_ldm, a_cur, (Mc * Kc) << 2, 0,
                                               &reply);
                    if (_ROW == xi)
                        athread_rma_col_ibcast(bb_ldm, b_cur, (Kc * Nc) << 2, 0,
                                               &reply);
                    athread_rma_wait_value(&reply, 2);
                    //                    sync_64spe();
                    athread_ssync_array();
                    ed = athread_stime_cycle();
                    commtime += ed - st;
                    // cal kernel
                    st = athread_stime_cycle();

                    for (i = 0; i < Mc; i++) {
                        for (k = 0; k < Kc; k++) {
                            for (j = 0; j < Nc; j++) {
                                c_ldm[i * Nc + j] +=
                                    aa_ldm[k * Mc + i] * bb_ldm[k * Nc + j];
                            }
                        }
                    }
                    ed = athread_stime_cycle();
                    caltime += ed - st;
                }
            }

            st = athread_stime_cycle();
            athread_ssync_array();
            reply = 0;
            athread_dma_iput_stride(&addr_c[STEP_C], c_ldm, (Mc * Nc) << 2,
                                    Nc << 2, (LDC - Nc) << 2, &reply);
            athread_dma_wait_value(&reply, 1);
            ed = athread_stime_cycle();
            commtime += ed - st;
        }
    }
}

void NN_384_0(swptex_mmPara *_) {
    unsigned long st, ed, allst, alled, caltime = 0, commtime = 0;
    allst = athread_stime_cycle();
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->K;
    int LDB = para->N;
    int LDC = para->N;
    int i, j, k, ii, jj, kk, k_end;

    // 传入的分块大小
    int Mc = para->Mc, Nc = para->Nc, Kc = para->Kc;
    int Ksize = para->Ksize;

    floatv8 va, vb, vc;
    float A_Part;

    for (ii = myid * Mc; ii < M; ii += Mc * thread_num) {
        for (jj = 0; jj < N; jj += Nc) {
            for (kk = 0; kk < Mc * Nc; kk++)
                c_ldm[kk] = 0.0;
            for (kk = 0; kk < K; kk += Kc) {
                reply = 0;
                st = athread_stime_cycle();

                athread_dma_iget_stride(a_ldm, &A[ii * LDA + kk],
                                        (Mc * Kc) << 2, Kc << 2,
                                        (LDA - Kc) << 2, &reply);
                athread_dma_iget_stride(b_ldm, &B[kk * LDB + jj],
                                        (Kc * Nc) << 2, Nc << 2,
                                        (LDB - Nc) << 2, &reply);
                athread_dma_wait_value(&reply, 2);
                ed = athread_stime_cycle();
                commtime += ed - st;

                st = athread_stime_cycle();
#if USE_KERNEL
                padding_kernel(a_ldm, b_ldm, c_ldm, Mc, Nc, Kc, Ksize);
#else
                for (i = 0; i < Mc; i++) {
                    for (k = 0; k < Kc; k++) {
                        A_Part = a_ldm[i * Kc + k];
                        va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part,
                                              A_Part, A_Part, A_Part, A_Part);
                        for (j = 0; j < Nc; j += 8) {
                            simd_load(vc, &c_ldm[i * Nc + j]);
                            simd_load(vb, &b_ldm[k * Nc + j]);
                            vc = simd_vmas(va, vb, vc);
                            simd_store(vc, &c_ldm[i * Nc + j]);
                        }
                    }
                }
#endif
                ed = athread_stime_cycle();
                caltime += ed - st;
            }
#if USE_KERNEL && !USE_FAST_SIMD
            for (i = 0; i < Mc; i++) {
                for (j = 0; j < Nc; j++) {
                    c_ldm[i * Nc + j] *= (-1.0);
                }
            }
#endif

            st = athread_stime_cycle();
            reply = 0;
            athread_dma_iput_stride(&C[ii * LDC + jj], c_ldm, (Mc * Nc) << 2,
                                    Nc << 2, (LDC - Nc) << 2, &reply);
            athread_dma_wait_value(&reply, 1);
            ed = athread_stime_cycle();
            commtime += ed - st;
        }
    }
#if SLAVE_LOG
    alled = athread_stime_cycle();
    if (myid == 0)
        printf("##calculate time: %.3f, communication time: %.3f, all time: "
               "%.3f\n",
               1.0 * caltime / 2100000, 1.0 * commtime / 2100000,
               1.0 * (alled - allst) / 2100000);
#endif
}

void NN_0_384(swptex_mmPara *_) {
    unsigned long st, ed, allst, alled, caltime = 0, commtime = 0;
    allst = athread_stime_cycle();
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->K;
    int LDB = para->N;
    int LDC = para->N;
    int i, j, k, ii, jj, kk, k_end;

    // 传入的分块大小
    int Mc = para->Mc, Nc = para->Nc, Kc = para->Kc;
    int Ksize = para->Ksize;

    floatv8 va, vb, vc;
    float A_Part;

    for (ii = 0; ii < M; ii += Mc) {
        for (jj = myid * Nc; jj < N; jj += Nc * thread_num) {
            for (kk = 0; kk < Mc * Nc; kk++)
                c_ldm[kk] = 0.0;
            for (kk = 0; kk < K; kk += Kc) {
                reply = 0;
                st = athread_stime_cycle();

                athread_dma_iget_stride(a_ldm, &A[ii * LDA + kk],
                                        (Mc * Kc) << 2, Kc << 2,
                                        (LDA - Kc) << 2, &reply);
                athread_dma_iget_stride(b_ldm, &B[kk * LDB + jj],
                                        (Kc * Nc) << 2, Nc << 2,
                                        (LDB - Nc) << 2, &reply);
                athread_dma_wait_value(&reply, 2);
                ed = athread_stime_cycle();
                commtime += ed - st;

                st = athread_stime_cycle();
#if USE_KERNEL
                padding_kernel(a_ldm, b_ldm, c_ldm, Mc, Nc, Kc, Ksize);
#else
                for (i = 0; i < Mc; i++) {
                    for (k = 0; k < Kc; k++) {
                        A_Part = a_ldm[i * Kc + k];
                        va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part,
                                              A_Part, A_Part, A_Part, A_Part);
                        for (j = 0; j < Nc; j += 8) {
                            simd_load(vc, &c_ldm[i * Nc + j]);
                            simd_load(vb, &b_ldm[k * Nc + j]);
                            vc = simd_vmas(va, vb, vc);
                            simd_store(vc, &c_ldm[i * Nc + j]);
                        }
                    }
                }
#endif
                ed = athread_stime_cycle();
                caltime += ed - st;
            }
#if USE_KERNEL && !USE_FAST_SIMD
            for (i = 0; i < Mc; i++) {
                for (j = 0; j < Nc; j++) {
                    c_ldm[i * Nc + j] *= (-1.0);
                }
            }
#endif

            st = athread_stime_cycle();
            reply = 0;
            athread_dma_iput_stride(&C[ii * LDC + jj], c_ldm, (Mc * Nc) << 2,
                                    Nc << 2, (LDC - Nc) << 2, &reply);
            athread_dma_wait_value(&reply, 1);
            ed = athread_stime_cycle();
            commtime += ed - st;
        }
    }
#if SLAVE_LOG
    alled = athread_stime_cycle();
    if (myid == 0)
        printf("##calculate time: %.3f, communication time: %.3f, all time: "
               "%.3f\n",
               1.0 * caltime / 2100000, 1.0 * commtime / 2100000,
               1.0 * (alled - allst) / 2100000);
#endif
}

void NN_6_64(swptex_mmPara *_) {
    unsigned long st, ed, allst, alled, caltime = 0, commtime = 0;
    allst = athread_stime_cycle();
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->K;
    int LDB = para->N;
    int LDC = para->N;
    int i, j, k, ii, jj, kk, k_end;

    // 传入的分块大小
    int Mc = para->Mc, Nc = para->Nc, Kc = para->Kc;
    int Ksize = para->Ksize;

    floatv8 va, vb, vc;
    float A_Part;

    for (ii = _CGN * Mc; ii < M; ii += Mc * 6) {
        for (jj = _PEN * Nc; jj < N; jj += Nc << 6) {
            for (kk = 0; kk < Mc * Nc; kk++)
                c_ldm[kk] = 0.0;
            for (kk = 0; kk < K; kk += Kc) {
                reply = 0;
                st = athread_stime_cycle();

                athread_dma_iget_stride(a_ldm, &A[ii * LDA + kk],
                                        (Mc * Kc) << 2, Kc << 2,
                                        (LDA - Kc) << 2, &reply);
                athread_dma_iget_stride(b_ldm, &B[kk * LDB + jj],
                                        (Kc * Nc) << 2, Nc << 2,
                                        (LDB - Nc) << 2, &reply);
                athread_dma_wait_value(&reply, 2);
                ed = athread_stime_cycle();
                commtime += ed - st;

                st = athread_stime_cycle();
#if USE_KERNEL
                padding_kernel(a_ldm, b_ldm, c_ldm, Mc, Nc, Kc, Ksize);
#else
                for (i = 0; i < Mc; i++) {
                    for (k = 0; k < Kc; k++) {
                        A_Part = a_ldm[i * Kc + k];
                        va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part,
                                              A_Part, A_Part, A_Part, A_Part);
                        for (j = 0; j < Nc; j += 8) {
                            simd_load(vc, &c_ldm[i * Nc + j]);
                            simd_load(vb, &b_ldm[k * Nc + j]);
                            vc = simd_vmas(va, vb, vc);
                            simd_store(vc, &c_ldm[i * Nc + j]);
                        }
                    }
                }
#endif
                ed = athread_stime_cycle();
                caltime += ed - st;
            }
#if USE_KERNEL && !USE_FAST_SIMD
            for (i = 0; i < Mc; i++) {
                for (j = 0; j < Nc; j++) {
                    c_ldm[i * Nc + j] *= (-1.0);
                }
            }
#endif

            st = athread_stime_cycle();
            reply = 0;
            athread_dma_iput_stride(&C[ii * LDC + jj], c_ldm, (Mc * Nc) << 2,
                                    Nc << 2, (LDC - Nc) << 2, &reply);
            athread_dma_wait_value(&reply, 1);
            ed = athread_stime_cycle();
            commtime += ed - st;
        }
    }
#if SLAVE_LOG
    alled = athread_stime_cycle();
    if (myid == 0)
        printf("##calculate time: %.3f, communication time: %.3f, all time: "
               "%.3f\n",
               1.0 * caltime / 2100000, 1.0 * commtime / 2100000,
               1.0 * (alled - allst) / 2100000);
#endif
}

void NN_64_6(swptex_mmPara *_) {
    unsigned long st, ed, allst, alled, caltime = 0, commtime = 0;
    allst = athread_stime_cycle();
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->K;
    int LDB = para->N;
    int LDC = para->N;
    int i, j, k, ii, jj, kk, k_end;

    // 传入的分块大小
    int Mc = para->Mc, Nc = para->Nc, Kc = para->Kc;
    int Ksize = para->Ksize;

    floatv8 va, vb, vc;
    float A_Part;

    for (ii = _PEN * Mc; ii < M; ii += Mc << 6) {
        for (jj = _CGN * Nc; jj < N; jj += Nc * 6) {
            for (kk = 0; kk < Mc * Nc; kk++)
                c_ldm[kk] = 0.0;
            for (kk = 0; kk < K; kk += Kc) {
                reply = 0;
                st = athread_stime_cycle();

                athread_dma_iget_stride(a_ldm, &A[ii * LDA + kk],
                                        (Mc * Kc) << 2, Kc << 2,
                                        (LDA - Kc) << 2, &reply);
                athread_dma_iget_stride(b_ldm, &B[kk * LDB + jj],
                                        (Kc * Nc) << 2, Nc << 2,
                                        (LDB - Nc) << 2, &reply);
                athread_dma_wait_value(&reply, 2);
                ed = athread_stime_cycle();
                commtime += ed - st;

                st = athread_stime_cycle();
#if USE_KERNEL
                padding_kernel(a_ldm, b_ldm, c_ldm, Mc, Nc, Kc, Ksize);
#else
                for (i = 0; i < Mc; i++) {
                    for (k = 0; k < Kc; k++) {
                        A_Part = a_ldm[i * Kc + k];
                        va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part,
                                              A_Part, A_Part, A_Part, A_Part);
                        for (j = 0; j < Nc; j += 8) {
                            simd_load(vc, &c_ldm[i * Nc + j]);
                            simd_load(vb, &b_ldm[k * Nc + j]);
                            vc = simd_vmas(va, vb, vc);
                            simd_store(vc, &c_ldm[i * Nc + j]);
                        }
                    }
                }
#endif
                ed = athread_stime_cycle();
                caltime += ed - st;
            }
#if USE_KERNEL && !USE_FAST_SIMD
            for (i = 0; i < Mc; i++) {
                for (j = 0; j < Nc; j++) {
                    c_ldm[i * Nc + j] *= (-1.0);
                }
            }
#endif

            st = athread_stime_cycle();
            reply = 0;
            athread_dma_iput_stride(&C[ii * LDC + jj], c_ldm, (Mc * Nc) << 2,
                                    Nc << 2, (LDC - Nc) << 2, &reply);
            athread_dma_wait_value(&reply, 1);
            ed = athread_stime_cycle();
            commtime += ed - st;
        }
    }
#if SLAVE_LOG
    alled = athread_stime_cycle();
    if (myid == 0)
        printf("##calculate time: %.3f, communication time: %.3f, all time: "
               "%.3f\n",
               1.0 * caltime / 2100000, 1.0 * commtime / 2100000,
               1.0 * (alled - allst) / 2100000);
#endif
}

void NT_384_0(swptex_mmPara *_) {
    unsigned long st, ed, allst, alled, caltime = 0, commtime = 0;
    allst = athread_stime_cycle();
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->K;
    int LDB = para->K;
    int LDC = para->N;
    int i, j, k, ii, jj, kk, k_end;
    // 传入的分块大小
    int Mc = para->Mc, Nc = para->Nc, Kc = para->Kc;
    int Ksize = para->Ksize;

    floatv8 va, vb, vc;
    float A_Part;

    for (ii = myid * Mc; ii < M; ii += Mc * thread_num) {
        for (jj = 0; jj < N; jj += Nc) {
            for (kk = 0; kk < Mc * Nc; kk++)
                c_ldm[kk] = 0.0;
            for (kk = 0; kk < K; kk += Kc) {
                reply = 0;
                st = athread_stime_cycle();

                athread_dma_iget_stride(a_ldm, &A[ii * LDA + kk],
                                        (Mc * Kc) << 2, Kc << 2,
                                        (LDA - Kc) << 2, &reply);
                athread_dma_iget_stride(b_ldm, &B[jj * LDB + kk],
                                        (Kc * Nc) << 2, Kc << 2,
                                        (LDB - Kc) << 2, &reply);
                athread_dma_wait_value(&reply, 2);
                ed = athread_stime_cycle();
                commtime += ed - st;

                st = athread_stime_cycle();
                //转置
                for (j = 0; j < Nc; j++) {
                    for (k = 0; k < Kc; k++) {
                        bb_ldm[k * Nc + j] = b_ldm[j * Kc + k];
                    }
                }
#if USE_KERNEL
                padding_kernel(a_ldm, bb_ldm, c_ldm, Mc, Nc, Kc, Ksize);
#else
                for (i = 0; i < Mc; i++) {
                    for (k = 0; k < Kc; k++) {
                        A_Part = a_ldm[i * Kc + k];
                        va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part,
                                              A_Part, A_Part, A_Part, A_Part);
                        for (j = 0; j < Nc; j += 8) {
                            simd_load(vc, &c_ldm[i * Nc + j]);
                            simd_load(vb, &bb_ldm[k * Nc + j]);
                            vc = simd_vmas(va, vb, vc);
                            simd_store(vc, &c_ldm[i * Nc + j]);
                        }
                    }
                }
#endif
                ed = athread_stime_cycle();
                caltime += ed - st;
            }
#if USE_KERNEL & !USE_FAST_SIMD
            for (i = 0; i < Mc; i++) {
                for (j = 0; j < Nc; j++) {
                    c_ldm[i * Nc + j] *= (-1.0);
                }
            }
#endif

            st = athread_stime_cycle();
            reply = 0;
            athread_dma_iput_stride(&C[ii * LDC + jj], c_ldm, (Mc * Nc) << 2,
                                    Nc << 2, (LDC - Nc) << 2, &reply);
            athread_dma_wait_value(&reply, 1);
            ed = athread_stime_cycle();
            commtime += ed - st;
        }
    }
    alled = athread_stime_cycle();

#if SLAVE_LOG
    if (myid == 0)
        printf("##calculate time: %.3f, communication time: %.3f, all time: "
               "%.3f\n",
               1.0 * caltime / 2100000, 1.0 * commtime / 2100000,
               1.0 * (alled - allst) / 2100000);
#endif
}

void NT_0_384(swptex_mmPara *_) {
    unsigned long st, ed, allst, alled, caltime = 0, commtime = 0;
    allst = athread_stime_cycle();
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->K;
    int LDB = para->K;
    int LDC = para->N;
    int i, j, k, ii, jj, kk, k_end;
    // 传入的分块大小
    int Mc = para->Mc, Nc = para->Nc, Kc = para->Kc;
    int Ksize = para->Ksize;

    floatv8 va, vb, vc;
    float A_Part;

    for (ii = 0; ii < M; ii += Mc) {
        for (jj = myid * Nc; jj < N; jj += Nc * thread_num) {
            for (kk = 0; kk < Mc * Nc; kk++)
                c_ldm[kk] = 0.0;
            for (kk = 0; kk < K; kk += Kc) {
                reply = 0;
                st = athread_stime_cycle();

                athread_dma_iget_stride(a_ldm, &A[ii * LDA + kk],
                                        (Mc * Kc) << 2, Kc << 2,
                                        (LDA - Kc) << 2, &reply);
                athread_dma_iget_stride(b_ldm, &B[jj * LDB + kk],
                                        (Kc * Nc) << 2, Kc << 2,
                                        (LDB - Kc) << 2, &reply);
                athread_dma_wait_value(&reply, 2);
                ed = athread_stime_cycle();
                commtime += ed - st;

                st = athread_stime_cycle();
                //转置
                for (j = 0; j < Nc; j++) {
                    for (k = 0; k < Kc; k++) {
                        bb_ldm[k * Nc + j] = b_ldm[j * Kc + k];
                    }
                }
#if USE_KERNEL
                padding_kernel(a_ldm, bb_ldm, c_ldm, Mc, Nc, Kc, Ksize);
#else
                for (i = 0; i < Mc; i++) {
                    for (k = 0; k < Kc; k++) {
                        A_Part = a_ldm[i * Kc + k];
                        va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part,
                                              A_Part, A_Part, A_Part, A_Part);
                        for (j = 0; j < Nc; j += 8) {
                            simd_load(vc, &c_ldm[i * Nc + j]);
                            simd_load(vb, &bb_ldm[k * Nc + j]);
                            vc = simd_vmas(va, vb, vc);
                            simd_store(vc, &c_ldm[i * Nc + j]);
                        }
                    }
                }
#endif
                ed = athread_stime_cycle();
                caltime += ed - st;
            }
#if USE_KERNEL & !USE_FAST_SIMD
            for (i = 0; i < Mc; i++) {
                for (j = 0; j < Nc; j++) {
                    c_ldm[i * Nc + j] *= (-1.0);
                }
            }
#endif

            st = athread_stime_cycle();
            reply = 0;
            athread_dma_iput_stride(&C[ii * LDC + jj], c_ldm, (Mc * Nc) << 2,
                                    Nc << 2, (LDC - Nc) << 2, &reply);
            athread_dma_wait_value(&reply, 1);
            ed = athread_stime_cycle();
            commtime += ed - st;
        }
    }
    alled = athread_stime_cycle();

#if SLAVE_LOG
    if (myid == 0)
        printf("##calculate time: %.3f, communication time: %.3f, all time: "
               "%.3f\n",
               1.0 * caltime / 2100000, 1.0 * commtime / 2100000,
               1.0 * (alled - allst) / 2100000);
#endif
}

void NT_64_6(swptex_mmPara *_) {
    unsigned long st, ed, allst, alled, caltime = 0, commtime = 0;
    allst = athread_stime_cycle();
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->K;
    int LDB = para->K;
    int LDC = para->N;
    int i, j, k, ii, jj, kk, k_end;
    // 传入的分块大小
    int Mc = para->Mc, Nc = para->Nc, Kc = para->Kc;
    int Ksize = para->Ksize;

    floatv8 va, vb, vc;
    float A_Part;

    for (ii = _PEN * Mc; ii < M; ii += Mc << 6) {
        for (jj = _CGN * Nc; jj < N; jj += Nc * 6) {
            for (kk = 0; kk < Mc * Nc; kk++)
                c_ldm[kk] = 0.0;
            for (kk = 0; kk < K; kk += Kc) {
                reply = 0;
                st = athread_stime_cycle();

                athread_dma_iget_stride(a_ldm, &A[ii * LDA + kk],
                                        (Mc * Kc) << 2, Kc << 2,
                                        (LDA - Kc) << 2, &reply);
                athread_dma_iget_stride(b_ldm, &B[jj * LDB + kk],
                                        (Kc * Nc) << 2, Kc << 2,
                                        (LDB - Kc) << 2, &reply);
                athread_dma_wait_value(&reply, 2);
                ed = athread_stime_cycle();
                commtime += ed - st;

                st = athread_stime_cycle();
                //转置
                for (j = 0; j < Nc; j++) {
                    for (k = 0; k < Kc; k++) {
                        bb_ldm[k * Nc + j] = b_ldm[j * Kc + k];
                    }
                }
#if USE_KERNEL
                padding_kernel(a_ldm, bb_ldm, c_ldm, Mc, Nc, Kc, Ksize);
#else
                for (i = 0; i < Mc; i++) {
                    for (k = 0; k < Kc; k++) {
                        A_Part = a_ldm[i * Kc + k];
                        va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part,
                                              A_Part, A_Part, A_Part, A_Part);
                        for (j = 0; j < Nc; j += 8) {
                            simd_load(vc, &c_ldm[i * Nc + j]);
                            simd_load(vb, &bb_ldm[k * Nc + j]);
                            vc = simd_vmas(va, vb, vc);
                            simd_store(vc, &c_ldm[i * Nc + j]);
                        }
                    }
                }
#endif
                ed = athread_stime_cycle();
                caltime += ed - st;
            }
#if USE_KERNEL & !USE_FAST_SIMD
            for (i = 0; i < Mc; i++) {
                for (j = 0; j < Nc; j++) {
                    c_ldm[i * Nc + j] *= (-1.0);
                }
            }
#endif

            st = athread_stime_cycle();
            reply = 0;
            athread_dma_iput_stride(&C[ii * LDC + jj], c_ldm, (Mc * Nc) << 2,
                                    Nc << 2, (LDC - Nc) << 2, &reply);
            athread_dma_wait_value(&reply, 1);
            ed = athread_stime_cycle();
            commtime += ed - st;
        }
    }
    alled = athread_stime_cycle();

#if SLAVE_LOG
    if (myid == 0)
        printf("##calculate time: %.3f, communication time: %.3f, all time: "
               "%.3f\n",
               1.0 * caltime / 2100000, 1.0 * commtime / 2100000,
               1.0 * (alled - allst) / 2100000);
#endif
}

void NT_6_64(swptex_mmPara *_) {
    unsigned long st, ed, allst, alled, caltime = 0, commtime = 0;
    allst = athread_stime_cycle();
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->K;
    int LDB = para->K;
    int LDC = para->N;
    int i, j, k, ii, jj, kk, k_end;
    // 传入的分块大小
    int Mc = para->Mc, Nc = para->Nc, Kc = para->Kc;
    int Ksize = para->Ksize;

    floatv8 va, vb, vc;
    float A_Part;

    for (ii = _CGN * Mc; ii < M; ii += Mc * 6) {
        for (jj = _PEN * Nc; jj < N; jj += Nc << 6) {
            for (kk = 0; kk < Mc * Nc; kk++)
                c_ldm[kk] = 0.0;
            for (kk = 0; kk < K; kk += Kc) {
                reply = 0;
                st = athread_stime_cycle();

                athread_dma_iget_stride(a_ldm, &A[ii * LDA + kk],
                                        (Mc * Kc) << 2, Kc << 2,
                                        (LDA - Kc) << 2, &reply);
                athread_dma_iget_stride(b_ldm, &B[jj * LDB + kk],
                                        (Kc * Nc) << 2, Kc << 2,
                                        (LDB - Kc) << 2, &reply);
                athread_dma_wait_value(&reply, 2);
                ed = athread_stime_cycle();
                commtime += ed - st;

                st = athread_stime_cycle();
                //转置
                for (j = 0; j < Nc; j++) {
                    for (k = 0; k < Kc; k++) {
                        bb_ldm[k * Nc + j] = b_ldm[j * Kc + k];
                    }
                }
#if USE_KERNEL
                padding_kernel(a_ldm, bb_ldm, c_ldm, Mc, Nc, Kc, Ksize);
#else
                for (i = 0; i < Mc; i++) {
                    for (k = 0; k < Kc; k++) {
                        A_Part = a_ldm[i * Kc + k];
                        va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part,
                                              A_Part, A_Part, A_Part, A_Part);
                        for (j = 0; j < Nc; j += 8) {
                            simd_load(vc, &c_ldm[i * Nc + j]);
                            simd_load(vb, &bb_ldm[k * Nc + j]);
                            vc = simd_vmas(va, vb, vc);
                            simd_store(vc, &c_ldm[i * Nc + j]);
                        }
                    }
                }
#endif
                ed = athread_stime_cycle();
                caltime += ed - st;
            }
#if USE_KERNEL & !USE_FAST_SIMD
            for (i = 0; i < Mc; i++) {
                for (j = 0; j < Nc; j++) {
                    c_ldm[i * Nc + j] *= (-1.0);
                }
            }
#endif

            st = athread_stime_cycle();
            reply = 0;
            athread_dma_iput_stride(&C[ii * LDC + jj], c_ldm, (Mc * Nc) << 2,
                                    Nc << 2, (LDC - Nc) << 2, &reply);
            athread_dma_wait_value(&reply, 1);
            ed = athread_stime_cycle();
            commtime += ed - st;
        }
    }
    alled = athread_stime_cycle();

#if SLAVE_LOG
    if (myid == 0)
        printf("##calculate time: %.3f, communication time: %.3f, all time: "
               "%.3f\n",
               1.0 * caltime / 2100000, 1.0 * commtime / 2100000,
               1.0 * (alled - allst) / 2100000);
#endif
}

void TN_384_0(swptex_mmPara *_) {
    unsigned long st, ed, allst, alled, caltime = 0, commtime = 0;
    allst = athread_stime_cycle();
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->M;
    int LDB = para->N;
    int LDC = para->N;
    int i, j, k, ii, jj, kk;

    // 传入的分块大小
    int Mc = para->Mc, Nc = para->Nc, Kc = para->Kc;
    int Ksize = para->Ksize;

    floatv8 va, vb, vc;
    float A_Part;

    for (ii = myid * Mc; ii < M; ii += Mc * thread_num) {
        for (jj = 0; jj < N; jj += Nc) {
            for (kk = 0; kk < Mc * Nc; kk++)
                c_ldm[kk] = 0.0;
            for (kk = 0; kk < K; kk += Kc) {
                reply = 0;
                st = athread_stime_cycle();

                athread_dma_iget_stride(a_ldm, &A[kk * LDA + ii],
                                        (Mc * Kc) << 2, Mc << 2,
                                        (LDA - Mc) << 2, &reply);
                athread_dma_iget_stride(b_ldm, &B[kk * LDB + jj],
                                        (Kc * Nc) << 2, Nc << 2,
                                        (LDB - Nc) << 2, &reply);
                athread_dma_wait_value(&reply, 2);
                ed = athread_stime_cycle();
                commtime += ed - st;
                st = athread_stime_cycle();

                //转置
                for (k = 0; k < Kc; k++) {
                    for (i = 0; i < Mc; i++) {
                        aa_ldm[i * Kc + k] = a_ldm[k * Mc + i];
                    }
                }
#if USE_KERNEL
                padding_kernel(aa_ldm, b_ldm, c_ldm, Mc, Nc, Kc, Ksize);
#else
                for (i = 0; i < Mc; i++) {
                    for (k = 0; k < Kc; k++) {
                        A_Part = aa_ldm[i * Kc + k];
                        va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part,
                                              A_Part, A_Part, A_Part, A_Part);
                        for (j = 0; j < Nc; j += 8) {
                            simd_load(vc, &c_ldm[i * Nc + j]);
                            simd_load(vb, &b_ldm[k * Nc + j]);
                            vc = simd_vmas(va, vb, vc);
                            simd_store(vc, &c_ldm[i * Nc + j]);
                        }
                    }
                }
#endif
                ed = athread_stime_cycle();
                caltime += ed - st;
            }
#if USE_KERNEL && !USE_FAST_SIMD
            for (i = 0; i < Mc; i++) {
                for (j = 0; j < Nc; j++) {
                    c_ldm[i * Nc + j] *= (-1.0);
                }
            }
#endif
            st = athread_stime_cycle();
            reply = 0;
            athread_dma_iput_stride(&C[ii * LDC + jj], c_ldm, (Mc * Nc) << 2,
                                    Nc << 2, (LDC - Nc) << 2, &reply);
            athread_dma_wait_value(&reply, 1);
            ed = athread_stime_cycle();
            commtime += ed - st;
        }
    }
#if SLAVE_LOG
    alled = athread_stime_cycle();
    if (myid == 0)
        printf("##calculate time: %.3f, communication time: %.3f, all time: "
               "%.3f\n",
               1.0 * caltime / 2100000, 1.0 * commtime / 2100000,
               1.0 * (alled - allst) / 2100000);
#endif
}

void TN_0_384(swptex_mmPara *_) {
    unsigned long st, ed, allst, alled, caltime = 0, commtime = 0;
    allst = athread_stime_cycle();
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->M;
    int LDB = para->N;
    int LDC = para->N;
    int i, j, k, ii, jj, kk;

    // 传入的分块大小
    int Mc = para->Mc, Nc = para->Nc, Kc = para->Kc;
    int Ksize = para->Ksize;

    floatv8 va, vb, vc;
    float A_Part;

    for (ii = 0; ii < M; ii += Mc) {
        for (jj = myid * Nc; jj < N; jj += Nc * thread_num) {
            for (kk = 0; kk < Mc * Nc; kk++)
                c_ldm[kk] = 0.0;
            for (kk = 0; kk < K; kk += Kc) {
                reply = 0;
                st = athread_stime_cycle();

                athread_dma_iget_stride(a_ldm, &A[kk * LDA + ii],
                                        (Mc * Kc) << 2, Mc << 2,
                                        (LDA - Mc) << 2, &reply);
                athread_dma_iget_stride(b_ldm, &B[kk * LDB + jj],
                                        (Kc * Nc) << 2, Nc << 2,
                                        (LDB - Nc) << 2, &reply);
                athread_dma_wait_value(&reply, 2);
                ed = athread_stime_cycle();
                commtime += ed - st;
                st = athread_stime_cycle();

                //转置
                for (k = 0; k < Kc; k++) {
                    for (i = 0; i < Mc; i++) {
                        aa_ldm[i * Kc + k] = a_ldm[k * Mc + i];
                    }
                }
#if USE_KERNEL
                padding_kernel(aa_ldm, b_ldm, c_ldm, Mc, Nc, Kc, Ksize);
#else
                for (i = 0; i < Mc; i++) {
                    for (k = 0; k < Kc; k++) {
                        A_Part = aa_ldm[i * Kc + k];
                        va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part,
                                              A_Part, A_Part, A_Part, A_Part);
                        for (j = 0; j < Nc; j += 8) {
                            simd_load(vc, &c_ldm[i * Nc + j]);
                            simd_load(vb, &b_ldm[k * Nc + j]);
                            vc = simd_vmas(va, vb, vc);
                            simd_store(vc, &c_ldm[i * Nc + j]);
                        }
                    }
                }
#endif
                ed = athread_stime_cycle();
                caltime += ed - st;
            }
#if USE_KERNEL && !USE_FAST_SIMD
            for (i = 0; i < Mc; i++) {
                for (j = 0; j < Nc; j++) {
                    c_ldm[i * Nc + j] *= (-1.0);
                }
            }
#endif
            st = athread_stime_cycle();
            reply = 0;
            athread_dma_iput_stride(&C[ii * LDC + jj], c_ldm, (Mc * Nc) << 2,
                                    Nc << 2, (LDC - Nc) << 2, &reply);
            athread_dma_wait_value(&reply, 1);
            ed = athread_stime_cycle();
            commtime += ed - st;
        }
    }
#if SLAVE_LOG
    alled = athread_stime_cycle();
    if (myid == 0)
        printf("##calculate time: %.3f, communication time: %.3f, all time: "
               "%.3f\n",
               1.0 * caltime / 2100000, 1.0 * commtime / 2100000,
               1.0 * (alled - allst) / 2100000);
#endif
}

void TN_64_6(swptex_mmPara *_) {
    unsigned long st, ed, allst, alled, caltime = 0, commtime = 0;
    allst = athread_stime_cycle();
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->M;
    int LDB = para->N;
    int LDC = para->N;
    int i, j, k, ii, jj, kk;

    // 传入的分块大小
    int Mc = para->Mc, Nc = para->Nc, Kc = para->Kc;
    int Ksize = para->Ksize;

    floatv8 va, vb, vc;
    float A_Part;

    for (ii = _PEN * Mc; ii < M; ii += Mc << 6) {
        for (jj = _CGN * Nc; jj < N; jj += Nc * 6) {
            for (kk = 0; kk < Mc * Nc; kk++)
                c_ldm[kk] = 0.0;
            for (kk = 0; kk < K; kk += Kc) {
                reply = 0;
                st = athread_stime_cycle();

                athread_dma_iget_stride(a_ldm, &A[kk * LDA + ii],
                                        (Mc * Kc) << 2, Mc << 2,
                                        (LDA - Mc) << 2, &reply);
                athread_dma_iget_stride(b_ldm, &B[kk * LDB + jj],
                                        (Kc * Nc) << 2, Nc << 2,
                                        (LDB - Nc) << 2, &reply);
                athread_dma_wait_value(&reply, 2);
                ed = athread_stime_cycle();
                commtime += ed - st;
                st = athread_stime_cycle();

                //转置
                for (k = 0; k < Kc; k++) {
                    for (i = 0; i < Mc; i++) {
                        aa_ldm[i * Kc + k] = a_ldm[k * Mc + i];
                    }
                }
#if USE_KERNEL
                padding_kernel(aa_ldm, b_ldm, c_ldm, Mc, Nc, Kc, Ksize);
#else
                for (i = 0; i < Mc; i++) {
                    for (k = 0; k < Kc; k++) {
                        A_Part = aa_ldm[i * Kc + k];
                        va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part,
                                              A_Part, A_Part, A_Part, A_Part);
                        for (j = 0; j < Nc; j += 8) {
                            simd_load(vc, &c_ldm[i * Nc + j]);
                            simd_load(vb, &b_ldm[k * Nc + j]);
                            vc = simd_vmas(va, vb, vc);
                            simd_store(vc, &c_ldm[i * Nc + j]);
                        }
                    }
                }
#endif
                ed = athread_stime_cycle();
                caltime += ed - st;
            }
#if USE_KERNEL && !USE_FAST_SIMD
            for (i = 0; i < Mc; i++) {
                for (j = 0; j < Nc; j++) {
                    c_ldm[i * Nc + j] *= (-1.0);
                }
            }
#endif
            st = athread_stime_cycle();
            reply = 0;
            athread_dma_iput_stride(&C[ii * LDC + jj], c_ldm, (Mc * Nc) << 2,
                                    Nc << 2, (LDC - Nc) << 2, &reply);
            athread_dma_wait_value(&reply, 1);
            ed = athread_stime_cycle();
            commtime += ed - st;
        }
    }
#if SLAVE_LOG
    alled = athread_stime_cycle();
    if (myid == 0)
        printf("##calculate time: %.3f, communication time: %.3f, all time: "
               "%.3f\n",
               1.0 * caltime / 2100000, 1.0 * commtime / 2100000,
               1.0 * (alled - allst) / 2100000);
#endif
}

void TN_6_64(swptex_mmPara *_) {
    unsigned long st, ed, allst, alled, caltime = 0, commtime = 0;
    allst = athread_stime_cycle();
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->M;
    int LDB = para->N;
    int LDC = para->N;
    int i, j, k, ii, jj, kk;

    // 传入的分块大小
    int Mc = para->Mc, Nc = para->Nc, Kc = para->Kc;
    int Ksize = para->Ksize;

    floatv8 va, vb, vc;
    float A_Part;

    for (ii = _CGN * Mc; ii < M; ii += Mc * 6) {
        for (jj = _PEN * Nc; jj < N; jj += Nc << 6) {
            for (kk = 0; kk < Mc * Nc; kk++)
                c_ldm[kk] = 0.0;
            for (kk = 0; kk < K; kk += Kc) {
                reply = 0;
                st = athread_stime_cycle();

                athread_dma_iget_stride(a_ldm, &A[kk * LDA + ii],
                                        (Mc * Kc) << 2, Mc << 2,
                                        (LDA - Mc) << 2, &reply);
                athread_dma_iget_stride(b_ldm, &B[kk * LDB + jj],
                                        (Kc * Nc) << 2, Nc << 2,
                                        (LDB - Nc) << 2, &reply);
                athread_dma_wait_value(&reply, 2);
                ed = athread_stime_cycle();
                commtime += ed - st;
                st = athread_stime_cycle();

                //转置
                for (k = 0; k < Kc; k++) {
                    for (i = 0; i < Mc; i++) {
                        aa_ldm[i * Kc + k] = a_ldm[k * Mc + i];
                    }
                }
#if USE_KERNEL
                padding_kernel(aa_ldm, b_ldm, c_ldm, Mc, Nc, Kc, Ksize);
#else
                for (i = 0; i < Mc; i++) {
                    for (k = 0; k < Kc; k++) {
                        A_Part = aa_ldm[i * Kc + k];
                        va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part,
                                              A_Part, A_Part, A_Part, A_Part);
                        for (j = 0; j < Nc; j += 8) {
                            simd_load(vc, &c_ldm[i * Nc + j]);
                            simd_load(vb, &b_ldm[k * Nc + j]);
                            vc = simd_vmas(va, vb, vc);
                            simd_store(vc, &c_ldm[i * Nc + j]);
                        }
                    }
                }
#endif
                ed = athread_stime_cycle();
                caltime += ed - st;
            }
#if USE_KERNEL && !USE_FAST_SIMD
            for (i = 0; i < Mc; i++) {
                for (j = 0; j < Nc; j++) {
                    c_ldm[i * Nc + j] *= (-1.0);
                }
            }
#endif
            st = athread_stime_cycle();
            reply = 0;
            athread_dma_iput_stride(&C[ii * LDC + jj], c_ldm, (Mc * Nc) << 2,
                                    Nc << 2, (LDC - Nc) << 2, &reply);
            athread_dma_wait_value(&reply, 1);
            ed = athread_stime_cycle();
            commtime += ed - st;
        }
    }
#if SLAVE_LOG
    alled = athread_stime_cycle();
    if (myid == 0)
        printf("##calculate time: %.3f, communication time: %.3f, all time: "
               "%.3f\n",
               1.0 * caltime / 2100000, 1.0 * commtime / 2100000,
               1.0 * (alled - allst) / 2100000);
#endif
}

//批量矩阵乘AB：仅划分bn，汇编核心64x64，补零（M<=64, N<=64, K<=64）
void sw_slave_bmm_AB_v1(swptex_mmPara *_) {
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->K;
    int LDB = para->N;
    int LDC = para->N;
    int BN = para->bn;
    int i, j, k, bn;
    floatv8 va, vb, vc;

    for (bn = myid; bn < BN; bn += thread_num) {
        reply = 0;
        athread_dma_iget_stride(a_ldm, &A[bn * M * LDA], (M * LDA) << 2, 0, 0,
                                &reply);
        athread_dma_iget_stride(b_ldm, &B[bn * K * LDB], (K * LDB) << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 2);
#if BMM_KERNEL //是否扩展
        // 计算矩阵均扩展成64x64
        for (i = 0; i < Mc; i++) {
            for (j = 0; j < Nc; j++) {
                c_ldm[i * Nc + j] = 0.0;
                aa_ldm[i * Kc + j] = 0.0;
                bb_ldm[i * Nc + j] = 0.0;
            }
        }
        for (i = 0; i < M; i++) {
            for (k = 0; k < K; k++) {
                aa_ldm[i * Kc + k] = a_ldm[i * LDA + k];
            }
        }
        for (k = 0; k < K; k++) {
            for (j = 0; j < N; j++) {
                bb_ldm[k * Nc + j] = b_ldm[k * LDB + j];
            }
        }

        // calculate kernel_64x64
        kernel_vlenmas_nn_64x64_asm_fp0(aa_ldm, bb_ldm, c_ldm);
        //        for (i = 0; i < Mc; i++) {
        //            for (k = 0; k < Kc; k++) {
        //                for (j = 0; j < Nc; j++) {
        //                    c_ldm[i * Nc + j] -= aa_ldm[i * Kc + k] * bb_ldm[k
        //                    * Nc + j];
        //                }
        //            }
        //        }
        // 乘(-1)
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                c_ldm[i * LDC + j] = c_ldm[i * Nc + j] * (-1.0);
            }
        }
#else
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                c_ldm[i * LDC + j] = 0.0;
            }
        }
        for (i = 0; i < M; i++) {
            for (k = 0; k < K; k++) {
                for (j = 0; j < N; j++) {
                    c_ldm[i * LDC + j] +=
                        a_ldm[i * LDA + k] * b_ldm[k * LDB + j];
                }
            }
        }
#endif

        reply = 0;
        athread_dma_iput_stride(&C[bn * M * LDC], c_ldm, (M * LDC) << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);
    }
}

//批量矩阵乘ABT：仅划分bn，汇编核心64x64，补零（M<=64, N<=64, K<=64）
void sw_slave_bmm_ABT_v1(swptex_mmPara *_) {
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->K;
    int LDB = para->K;
    int LDC = para->N;
    int BN = para->bn;
    int i, j, k, bn;
    floatv8 va, vb, vc;
    float temp;

    for (bn = myid; bn < BN; bn += thread_num) {
        reply = 0;
        athread_dma_iget_stride(a_ldm, &A[bn * M * LDA], (M * LDA) << 2, 0, 0,
                                &reply);
        athread_dma_iget_stride(b_ldm, &B[bn * N * LDB], (N * LDB) << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 2);

#if BMM_KERNEL //是否扩展
        for (i = 0; i < Mc; i++) {
            for (j = 0; j < Nc; j++) {
                c_ldm[i * Nc + j] = 0.0;
                aa_ldm[i * Kc + j] = 0.0;
                bb_ldm[i * Nc + j] = 0.0;
            }
        }
        for (i = 0; i < M; i++) {
            for (k = 0; k < K; k++) {
                aa_ldm[i * Kc + k] = a_ldm[i * LDA + k];
            }
        }
        for (j = 0; j < N; j++) {
            for (k = 0; k < K; k++) {
                bb_ldm[k * Nc + j] = b_ldm[j * LDB + k];
            }
        }

        kernel_vlenmas_nn_64x64_asm_fp0(aa_ldm, bb_ldm, c_ldm);
        //        for (i = 0; i < Mc; i++) {
        //            for (k = 0; k < Kc; k++) {
        //                for (j = 0; j < Nc; j++) {
        //                    c_ldm[i * Nc + j] -= aa_ldm[i * Kc + k] * bb_ldm[k
        //                    * Nc + j];
        //                }
        //            }
        //        }
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                c_ldm[i * LDC + j] = c_ldm[i * Nc + j] * (-1.0);
            }
        }
#else
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                temp = 0.f;
                for (k = 0; k < K; k++) {
                    temp += a_ldm[i * LDA + k] * b_ldm[j * LDB + k];
                }
                c_ldm[i * LDC + j] = temp;
            }
        }
#endif
        reply = 0;
        athread_dma_iput_stride(&C[bn * M * LDC], c_ldm, (M * LDC) << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);
    }
}

//批量矩阵乘ATB：仅划分bn，汇编核心64x64，补零（M<=64, N<=64, K<=64）
void sw_slave_bmm_ATB_v1(swptex_mmPara *_) {
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int LDA = para->M;
    int LDB = para->N;
    int LDC = para->N;
    int BN = para->bn;
    int i, j, k, bn;
    floatv8 va, vb, vc;
    float temp;

    for (bn = myid; bn < BN; bn += thread_num) {
        reply = 0;
        athread_dma_iget_stride(a_ldm, &A[bn * K * LDA], (K * LDA) << 2, 0, 0,
                                &reply);
        athread_dma_iget_stride(b_ldm, &B[bn * K * LDB], (K * LDB) << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 2);

#if BMM_KERNEL //是否扩展
        for (i = 0; i < Mc; i++) {
            for (j = 0; j < Nc; j++) {
                c_ldm[i * Nc + j] = 0.0;
                aa_ldm[i * Kc + j] = 0.0;
                bb_ldm[i * Nc + j] = 0.0;
            }
        }
        for (i = 0; i < M; i++) {
            for (k = 0; k < K; k++) {
                aa_ldm[i * Kc + k] = a_ldm[k * LDA + i];
            }
        }
        for (j = 0; j < N; j++) {
            for (k = 0; k < K; k++) {
                bb_ldm[k * Nc + j] = b_ldm[k * LDB + j];
            }
        }

        kernel_vlenmas_nn_64x64_asm_fp0(aa_ldm, bb_ldm, c_ldm);

        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                c_ldm[i * LDC + j] = c_ldm[i * Nc + j] * (-1.0);
            }
        }
#else
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                temp = 0.f;
                for (k = 0; k < K; k++) {
                    temp += a_ldm[k * LDA + i] * b_ldm[k * LDB + j];
                }
                c_ldm[i * LDC + j] = temp;
            }
        }
#endif

        reply = 0;
        athread_dma_iput_stride(&C[bn * M * LDC], c_ldm, (M * LDC) << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);
    }
}

inline double fast_exp(double y) {
    // double y = (double)y_;
    union {
        struct {
            uint32_t i;
            uint32_t j;
        } n;
        double d;
    } v;
    v.n.j = (1 << 20) * (1.4426950409 * y + 1022.9420151853);
    return v.d;
}

void sw_softmax(swptex_mmPara *_) {
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    float *x = para->x;
    size_t M = para->M;
    size_t N = para->N;
    int i, j;

    int myid = CRTS_cgn * 64 + CRTS_tid;

    float tmp, sum;

    for (i = myid; i < M; i += thread_num) {

        reply = 0;
        athread_dma_iget_stride(x_ldm, &x[i * N], N << 2, 0, 0, &reply);
        athread_dma_wait_value(&reply, 1);
        tmp = x_ldm[0];
        for (j = 1; j < N; ++j) {
            if (x_ldm[j] > tmp) {
                tmp = x_ldm[j];
            }
        }
        sum = 0.f;
        // if(myid==0){
        //     printf("%f %f %f\n",x_ldm[0] - tmp,fast_exp(x_ldm[0] -
        //     tmp),exp(x_ldm[0] - tmp));
        // }
        for (j = 0; j < N; ++j) {
            x_ldm[j] = exp(x_ldm[j] - tmp);
            sum += x_ldm[j];
        }
        for (j = 0; j < N; ++j) {
            x_ldm[j] /= sum;
        }

        reply = 0;
        athread_dma_iput_stride(&x[i * N], x_ldm, N << 2, 0, 0, &reply);
        athread_dma_wait_value(&reply, 1);
    }
}

void sw_dsoftmax(swptex_mmPara *_) {
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    float *dy = para->dy;
    float *y = para->y;
    size_t M = para->M;
    size_t N = para->N;
    int i, j;

    int myid = CRTS_cgn * 64 + CRTS_tid;

    float tmp;
    for (i = myid; i < M; i += thread_num) {
        tmp = 0.f;

        reply = 0;
        athread_dma_iget_stride(dy_ldm, &dy[i * N], N << 2, 0, 0, &reply);
        athread_dma_iget_stride(y_ldm, &y[i * N], N << 2, 0, 0, &reply);
        athread_dma_wait_value(&reply, 2);

        for (j = 0; j < N; ++j) {
            tmp += dy_ldm[j] * y_ldm[j];
        }

        for (j = 0; j < N; ++j) {
            dy_ldm[j] = (dy_ldm[j] - tmp) * y_ldm[j];
        }

        reply = 0;
        athread_dma_iput_stride(&dy[i * N], dy_ldm, N << 2, 0, 0, &reply);
        athread_dma_wait_value(&reply, 1);
    }
}

void sw_split_and_transpose(swptex_mmPara *_) {
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    float *QKV = para->QKV_data;
    float *Q = para->Q_data;
    float *K = para->K_data;
    float *V = para->V_data;

    int B = para->b, N = para->n, S = para->s, D = para->d;
    int myid = CRTS_cgn * 64 + CRTS_tid;

    int b, n, s;

    for (b = myid; b < B; b += thread_num) {
        for (n = 0; n < N; ++n) {
            for (s = 0; s < S; ++s) {
                float *addr_src_q =
                    QKV + n * D + s * N * D * 3 + b * S * N * D * 3;
                float *addr_src_k =
                    QKV + N * D + n * D + s * N * D * 3 + b * S * N * D * 3;
                float *addr_src_v =
                    QKV + N * D * 2 + n * D + s * N * D * 3 + b * S * N * D * 3;

                reply = 0;
                athread_dma_iget_stride(q_ldm, addr_src_q, D << 2, 0, 0,
                                        &reply);
                athread_dma_iget_stride(k_ldm, addr_src_k, D << 2, 0, 0,
                                        &reply);
                athread_dma_iget_stride(v_ldm, addr_src_v, D << 2, 0, 0,
                                        &reply);
                athread_dma_wait_value(&reply, 3);
                float *addr_dest_q = Q + b * N * S * D + n * S * D + s * D;
                float *addr_dest_k = K + b * N * S * D + n * S * D + s * D;
                float *addr_dest_v = V + b * N * S * D + n * S * D + s * D;

                reply = 0;
                athread_dma_iput_stride(addr_dest_q, q_ldm, D << 2, 0, 0,
                                        &reply);
                athread_dma_iput_stride(addr_dest_k, k_ldm, D << 2, 0, 0,
                                        &reply);
                athread_dma_iput_stride(addr_dest_v, v_ldm, D << 2, 0, 0,
                                        &reply);
                athread_dma_wait_value(&reply, 3);
            }
        }
    }
}

void sw_transpose_and_merge(swptex_mmPara *_) {
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    float *QKV = para->QKV_data;
    float *Q = para->Q_data;
    float *K = para->K_data;
    float *V = para->V_data;

    int B = para->b, N = para->n, S = para->s, D = para->d;
    int myid = CRTS_cgn * 64 + CRTS_tid;

    int b, n, s;
    for (b = myid; b < B; b += thread_num) {
        for (n = 0; n < N; ++n) {
            for (s = 0; s < S; ++s) {
                float *addr_src_q = Q + b * N * S * D + n * S * D + s * D;
                float *addr_src_k = K + b * N * S * D + n * S * D + s * D;
                float *addr_src_v = V + b * N * S * D + n * S * D + s * D;

                reply = 0;

                athread_dma_iget_stride(q_ldm, addr_src_q, D << 2, 0, 0,
                                        &reply);
                athread_dma_iget_stride(k_ldm, addr_src_k, D << 2, 0, 0,
                                        &reply);
                athread_dma_iget_stride(v_ldm, addr_src_v, D << 2, 0, 0,
                                        &reply);
                athread_dma_wait_value(&reply, 3);

                float *addr_dest_q =
                    QKV + n * D + s * N * D * 3 + b * S * N * D * 3;
                float *addr_dest_k =
                    QKV + N * D + n * D + s * N * D * 3 + b * S * N * D * 3;
                float *addr_dest_v =
                    QKV + N * D * 2 + n * D + s * N * D * 3 + b * S * N * D * 3;

                reply = 0;
                athread_dma_iput_stride(addr_dest_q, q_ldm, D << 2, 0, 0,
                                        &reply);
                athread_dma_iput_stride(addr_dest_k, k_ldm, D << 2, 0, 0,
                                        &reply);
                athread_dma_iput_stride(addr_dest_v, v_ldm, D << 2, 0, 0,
                                        &reply);
                athread_dma_wait_value(&reply, 3);
            }
        }
    }
}

void sw_split(swptex_mmPara *_) {
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    float *QKV = para->QKV_data;
    float *QKVT = para->QKVT_data;

    int B = para->b, N = para->n, S = para->s, D = para->d;
    int myid = CRTS_cgn * 64 + CRTS_tid;

    int b, n, s;
    for (b = myid; b < B; b += thread_num) {
        for (n = 0; n < N; ++n) {
            for (s = 0; s < S; ++s) {

                float *addr_src_qkv = QKV + n * D + s * N * D + b * S * N * D;

                reply = 0;
                athread_dma_iget_stride(q_ldm, addr_src_qkv, D << 2, 0, 0,
                                        &reply);
                athread_dma_wait_value(&reply, 1);

                float *addr_dest_qkv = QKVT + b * N * S * D + n * S * D + s * D;

                reply = 0;
                athread_dma_iput_stride(addr_dest_qkv, q_ldm, D << 2, 0, 0,
                                        &reply);
                athread_dma_wait_value(&reply, 1);
            }
        }
    }
}

void sw_merge(swptex_mmPara *_) {
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    float *QKV = para->QKV_data;
    float *QKVT = para->QKVT_data;

    int B = para->b, N = para->n, S = para->s, D = para->d;
    int myid = CRTS_cgn * 64 + CRTS_tid;

    int b, n, s;
    for (b = myid; b < B; b += thread_num) {
        for (n = 0; n < N; ++n) {
            for (s = 0; s < S; ++s) {

                float *addr_src_qkv = QKV + b * N * S * D + n * S * D + s * D;

                reply = 0;
                athread_dma_iget_stride(q_ldm, addr_src_qkv, D << 2, 0, 0,
                                        &reply);
                athread_dma_wait_value(&reply, 1);

                float *addr_dest_qkv = QKVT + n * D + s * N * D + b * S * N * D;

                reply = 0;
                athread_dma_iput_stride(addr_dest_qkv, q_ldm, D << 2, 0, 0,
                                        &reply);
                athread_dma_wait_value(&reply, 1);
            }
        }
    }
}

void sw_scale(swptex_mmPara *_) {
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    float *x = para->x;
    int len = para->len;
    float scaling = para->scaling;

    int myid = CRTS_cgn * 64 + CRTS_tid;
    int slaveProcessNum = len / (thread_num);
    int segNum = 1024;
    int seg = slaveProcessNum / segNum;
    int segTotal = seg * segNum;
    int location = myid * slaveProcessNum;

    int i, j;
    for (i = 0; i < segTotal; i += segNum) {
        reply = 0;
        athread_dma_iget_stride(x_ldm, &x[location + i], segNum << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);

        for (j = 0; j < segNum; j++) {
            x_ldm[j] *= scaling;
        }

        reply = 0;
        athread_dma_iput_stride(&x[location + i], x_ldm, segNum << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);
    }

    if (i < slaveProcessNum) {
        int num = slaveProcessNum - i;

        reply = 0;
        athread_dma_iget_stride(x_ldm, &x[location + i], num << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);

        for (j = 0; j < num; j++) {
            x_ldm[j] *= scaling;
        }

        reply = 0;
        athread_dma_iput_stride(&x[location + i], x_ldm, num << 2, 0, 0,
                                &reply);
        athread_dma_wait_value(&reply, 1);
    }

    int tail = len % thread_num;

    if (myid == 0 && tail != 0) {
        reply = 0;
        athread_dma_iget_stride(x_ldm, &x[len - tail], tail << 2, 0, 0, &reply);
        athread_dma_wait_value(&reply, 1);

        for (j = 0; j < tail; j++) {
            x_ldm[j] *= scaling;
        }

        athread_dma_iput_stride(&x[len - tail], x_ldm, tail << 2, 0, 0, &reply);
        athread_dma_wait_value(&reply, 1);
    }
}

void sw_sum_axis0(swptex_mmPara *_) {
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    float *src = para->x;
    float *dst = para->y;
    size_t M = para->M;
    size_t N = para->N;

    int myid = CRTS_cgn * 64 + CRTS_tid;

    int segNum = 1024;
    int seg = M / segNum;
    int segTotal = seg * segNum;

    int i, j;

    for (i = myid; i < N; i += thread_num) {
        dst_ldm[0] = 0.0;
        for (j = 0; j < segTotal; j += segNum) {
            reply = 0;
            athread_dma_iget_stride(x_ldm, &src[j * N + i], segNum << 2, 1 << 2,
                                    (N - 1) << 2, &reply);
            athread_dma_wait_value(&reply, 1);

            for (int k = 0; k < segNum; k++) {
                dst_ldm[0] += x_ldm[k];
            }
        }

        if (j < M) {
            int num = M - segTotal;
            reply = 0;
            athread_dma_iget_stride(x_ldm, &src[j * N + i], num << 2, 1 << 2,
                                    (N - 1) << 2, &reply);
            athread_dma_wait_value(&reply, 1);

            for (int k = 0; k < num; k++) {
                dst_ldm[0] += x_ldm[k];
            }
        }

        reply = 0;
        athread_dma_iput_stride(&dst[i], dst_ldm, 1 << 2, 0, 0, &reply);
        athread_dma_wait_value(&reply, 1);
    }
}

void sw_slave_mm_ABT(swptex_mmPara *_) {
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;

    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int i, j, k;
    float temp;
    int M_blk = M / thread_num;
    int M_load = M_blk + (myid < M % thread_num ? 1 : 0);
    int M_st = myid * M_blk + (myid < M % thread_num ? myid : M % thread_num);
    float *local_A = A + M_st * K;
    float *local_C = C + M_st * N;
    for (i = 0; i < M_load; ++i) {
        for (j = 0; j < N; ++j) {
            temp = 0.f;
            for (k = 0; k < K; ++k) {
                temp += local_A[i * K + k] * B[j * K + k];
            }
            local_C[i * N + j] = temp;
        }
    }
}

void sw_slave_mm_AB(swptex_mmPara *_) {
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int i, j, k;
    float temp;
    int M_blk = M / thread_num;
    int M_load = M_blk + (myid < M % thread_num ? 1 : 0);
    int M_st = myid * M_blk + (myid < M % thread_num ? myid : M % thread_num);
    float *local_A = A + M_st * K;
    float *local_C = C + M_st * N;
    for (i = 0; i < M_load; ++i) {
        for (j = 0; j < N; ++j) {
            temp = 0.f;
            for (k = 0; k < K; ++k) {
                temp += local_A[i * K + k] * B[k * N + j];
            }
            local_C[i * N + j] = temp;
        }
    }
}

void sw_slave_mm_ATB(swptex_mmPara *_) {
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int i, j, k;
    float temp;
    int M_blk = M / thread_num;
    int M_load = M_blk + (myid < M % thread_num ? 1 : 0);
    int M_st = myid * M_blk + (myid < M % thread_num ? myid : M % thread_num);
    for (i = 0; i < M_load; ++i) {
        for (j = 0; j < N; ++j) {
            temp = 0.f;
            for (k = 0; k < K; ++k) {
                temp += A[M_st + k * M + i] * B[k * N + j];
            }
            C[i * N + M_st * N + j] = temp;
        }
    }
}
void kernel_vlenmas_nn_64x64_asm_fp0(float *a_ldm, float *b_ldm, float *d_ldm) {
#if 0
    int i, j, k;
    floatv8 va, vb, vc;
    for(i=0; i<64; i++)
    for(k=0; k<64; k++){
        float A_Part = -1.0*a_ldm[i*64+k];
        va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part, A_Part, A_Part, A_Part, A_Part);
        for(j=0; j<64; j+=8){
            simd_load(vc, &d_ldm[i*64+j]);
            simd_load(vb, &b_ldm[k*64+j]);
            vc = simd_vmas(va, vb, vc);
            simd_store(vc, &d_ldm[i*64+j]);
        }
    }
#else
    int i, j, k, reg;
    doublev8 va, vb, vc, vaa, vcc;
    asm volatile("pws");
    asm volatile("ldi  %0, 0x20($31)\n"
                 "wcsr %0, 0x91\n"
                 "ldi  %0, 0xf($31)\n"
                 "wcsr %0, 0x92\n"
                 "ldi  %0, 0x0(%1)\n"
                 "wcsr %0, 0x90\n"
                 : "=&r"(reg)
                 : "r"(d_ldm));
    // asm("pws\n");
    for (i = 0; i < 64; i += 2) {
        asm volatile("vlds $47, 0x0(%8)\n"
                     "vlds $63, 0x20*1(%8)\n"
                     "vlds $60, 0x20*2(%8)\n"
                     "vlds $59, 0x20*3(%8)\n"
                     "vlds $57, 0x20*4(%8)\n"
                     "vlds $55, 0x20*5(%8)\n"
                     "vlds $54, 0x20*6(%8)\n"
                     "vlds $53, 0x20*7(%8)\n"
                     "vlds $44, (0x100+0x20*0)(%8)\n"
                     "vlds $43, (0x100+0x20*1)(%8)\n"
                     "vlds $42, (0x100+0x20*2)(%8)\n"
                     "vlds $41, (0x100+0x20*3)(%8)\n"
                     "vlds $39, (0x100+0x20*4)(%8)\n"
                     "vlds $38, (0x100+0x20*5)(%8)\n"
                     "vlds $37, (0x100+0x20*6)(%8)\n"
                     "vlds $36, (0x100+0x20*7)(%8)\n"
                     "vlds $46, 0x0(%6)\n"
                     "vlds $33, 0x100(%6)\n"
                     "vlds $45,0x0(%7)\n"
                     "vlds $62,0x20*1(%7)\n"
                     "vlds $61,0x20*2(%7)\n"
                     "vlds $58,0x20*3(%7)\n"
                     "vlds $52,0x20*4(%7)\n"
                     "vlds $51,0x20*5(%7)\n"
                     "vlds $50,0x20*6(%7)\n"
                     "vlds $49,0x20*7(%7)\n"
                     : "=&r"(reg), "=&r"(va), "=&r"(vb), "=&r"(vc), "=&r"(vaa),
                       "=&r"(vcc)
                     : "r"(&a_ldm[(i << 6) + 0]), "r"(&b_ldm[(0 << 6) + 0]),
                       "r"(&d_ldm[(i << 6) + 0]));
        for (k = 0; k < 56; k += 8) {
            asm volatile(
                //#0
                "vlenma $46, $45, $47, $47\n"
                "vlenma $33, $45, $44, $44\n"
                "vlds $45, 1*0x100(%7)\n"
                "vlenma $46, $62, $63, $63\n"
                "vlenma $33, $62, $43, $43\n"
                "vlds $62, (1*0x100+0x20*1)(%7)\n"
                "vlenma $46, $61, $60, $60\n"
                "vlenma $33, $61, $42, $42\n"
                "vlds $61, (1*0x100+0x20*2)(%7)\n"
                "vlenma $46, $58, $59, $59\n"
                "vlenma $33, $58, $41, $41\n"
                "vlds $58, (1*0x100+0x20*3)(%7)\n"
                "vlenma $46, $52, $57, $57\n"
                "vlenma $33, $52, $39, $39\n"
                "vlds $52, (1*0x100+0x20*4)(%7)\n"
                "vlenma $46, $51, $55, $55\n"
                "vlenma $33, $51, $38, $38\n"
                "vlds $51, (1*0x100+0x20*5)(%7)\n"
                "vlenma $46, $50, $54, $54\n"
                "vlenma $33, $50, $37, $37\n"
                "vlds $50, (1*0x100+0x20*6)(%7)\n"
                "vlenma $46, $49, $53, $53\n"
                "vlenma $33, $49, $36, $36\n"
                //#1
                "vlds $49, (1*0x100+0x20*7)(%7)\n"
                "vlenma $46, $45, $47, $47\n"
                "vlenma $33, $45, $44, $44\n"
                "vlds $45, 2*0x100(%7)\n"
                "vlenma $46, $62, $63, $63\n"
                "vlenma $33, $62, $43, $43\n"
                "vlds $62, (2*0x100+0x20*1)(%7)\n"
                "vlenma $46, $61, $60, $60\n"
                "vlenma $33, $61, $42, $42\n"
                "vlds $61, (2*0x100+0x20*2)(%7)\n"
                "vlenma $46, $58, $59, $59\n"
                "vlenma $33, $58, $41, $41\n"
                "vlds $58, (2*0x100+0x20*3)(%7)\n"
                "vlenma $46, $52, $57, $57\n"
                "vlenma $33, $52, $39, $39\n"
                "vlds $52, (2*0x100+0x20*4)(%7)\n"
                "vlenma $46, $51, $55, $55\n"
                "vlenma $33, $51, $38, $38\n"
                "vlds $51, (2*0x100+0x20*5)(%7)\n"
                "vlenma $46, $50, $54, $54\n"
                "vlenma $33, $50, $37, $37\n"
                "vlds $50, (2*0x100+0x20*6)(%7)\n"
                "vlenma $46, $49, $53, $53\n"
                "vlenma $33, $49, $36, $36\n"
                //#2
                "vlds $49, (2*0x100+0x20*7)(%7)\n"
                "vlenma $46, $45, $47, $47\n"
                "vlenma $33, $45, $44, $44\n"
                "vlds $45, 3*0x100(%7)\n"
                "vlenma $46, $62, $63, $63\n"
                "vlenma $33, $62, $43, $43\n"
                "vlds $62, (3*0x100+0x20*1)(%7)\n"
                "vlenma $46, $61, $60, $60\n"
                "vlenma $33, $61, $42, $42\n"
                "vlds $61, (3*0x100+0x20*2)(%7)\n"
                "vlenma $46, $58, $59, $59\n"
                "vlenma $33, $58, $41, $41\n"
                "vlds $58, (3*0x100+0x20*3)(%7)\n"
                "vlenma $46, $52, $57, $57\n"
                "vlenma $33, $52, $39, $39\n"
                "vlds $52, (3*0x100+0x20*4)(%7)\n"
                "vlenma $46, $51, $55, $55\n"
                "vlenma $33, $51, $38, $38\n"
                "vlds $51, (3*0x100+0x20*5)(%7)\n"
                "vlenma $46, $50, $54, $54\n"
                "vlenma $33, $50, $37, $37\n"
                "vlds $50, (3*0x100+0x20*6)(%7)\n"
                "vlenma $46, $49, $53, $53\n"
                "vlenma $33, $49, $36, $36\n"
                //#3
                "vlds $49, (3*0x100+0x20*7)(%7)\n"
                "vlenma $46, $45, $47, $47\n"
                "vlenma $33, $45, $44, $44\n"
                "vlds $45, 4*0x100(%7)\n"
                "vlenma $46, $62, $63, $63\n"
                "vlenma $33, $62, $43, $43\n"
                "vlds $62, (4*0x100+0x20*1)(%7)\n"
                "vlenma $46, $61, $60, $60\n"
                "vlenma $33, $61, $42, $42\n"
                "vlds $61, (4*0x100+0x20*2)(%7)\n"
                "vlenma $46, $58, $59, $59\n"
                "vlenma $33, $58, $41, $41\n"
                "vlds $58, (4*0x100+0x20*3)(%7)\n"
                "vlenma $46, $52, $57, $57\n"
                "vlenma $33, $52, $39, $39\n"
                "vlds $52, (4*0x100+0x20*4)(%7)\n"
                "vlenma $46, $51, $55, $55\n"
                "vlenma $33, $51, $38, $38\n"
                "vlds $51, (4*0x100+0x20*5)(%7)\n"
                "vlenma $46, $50, $54, $54\n"
                "vlenma $33, $50, $37, $37\n"
                "vlds $50, (4*0x100+0x20*6)(%7)\n"
                "vlenma $46, $49, $53, $53\n"
                "vlenma $33, $49, $36, $36\n"
                //#4
                "vlds $49, (4*0x100+0x20*7)(%7)\n"
                "vlenma $46, $45, $47, $47\n"
                "vlenma $33, $45, $44, $44\n"
                "vlds $45, 5*0x100(%7)\n"
                "vlenma $46, $62, $63, $63\n"
                "vlenma $33, $62, $43, $43\n"
                "vlds $62, (5*0x100+0x20*1)(%7)\n"
                "vlenma $46, $61, $60, $60\n"
                "vlenma $33, $61, $42, $42\n"
                "vlds $61, (5*0x100+0x20*2)(%7)\n"
                "vlenma $46, $58, $59, $59\n"
                "vlenma $33, $58, $41, $41\n"
                "vlds $58, (5*0x100+0x20*3)(%7)\n"
                "vlenma $46, $52, $57, $57\n"
                "vlenma $33, $52, $39, $39\n"
                "vlds $52, (5*0x100+0x20*4)(%7)\n"
                "vlenma $46, $51, $55, $55\n"
                "vlenma $33, $51, $38, $38\n"
                "vlds $51, (5*0x100+0x20*5)(%7)\n"
                "vlenma $46, $50, $54, $54\n"
                "vlenma $33, $50, $37, $37\n"
                "vlds $50, (5*0x100+0x20*6)(%7)\n"
                "vlenma $46, $49, $53, $53\n"
                "vlenma $33, $49, $36, $36\n"
                //#5
                "vlds $49, (5*0x100+0x20*7)(%7)\n"
                "vlenma $46, $45, $47, $47\n"
                "vlenma $33, $45, $44, $44\n"
                "vlds $45, 6*0x100(%7)\n"
                "vlenma $46, $62, $63, $63\n"
                "vlenma $33, $62, $43, $43\n"
                "vlds $62, (6*0x100+0x20*1)(%7)\n"
                "vlenma $46, $61, $60, $60\n"
                "vlenma $33, $61, $42, $42\n"
                "vlds $61, (6*0x100+0x20*2)(%7)\n"
                "vlenma $46, $58, $59, $59\n"
                "vlenma $33, $58, $41, $41\n"
                "vlds $58, (6*0x100+0x20*3)(%7)\n"
                "vlenma $46, $52, $57, $57\n"
                "vlenma $33, $52, $39, $39\n"
                "vlds $52, (6*0x100+0x20*4)(%7)\n"
                "vlenma $46, $51, $55, $55\n"
                "vlenma $33, $51, $38, $38\n"
                "vlds $51, (6*0x100+0x20*5)(%7)\n"
                "vlenma $46, $50, $54, $54\n"
                "vlenma $33, $50, $37, $37\n"
                "vlds $50, (6*0x100+0x20*6)(%7)\n"
                "vlenma $46, $49, $53, $53\n"
                "vlenma $33, $49, $36, $36\n"
                //#6
                "vlds $49, (6*0x100+0x20*7)(%7)\n"
                "vlenma $46, $45, $47, $47\n"
                "vlenma $33, $45, $44, $44\n"
                "vlds $45, 7*0x100(%7)\n"
                "vlenma $46, $62, $63, $63\n"
                "vlenma $33, $62, $43, $43\n"
                "vlds $62, (7*0x100+0x20*1)(%7)\n"
                "vlenma $46, $61, $60, $60\n"
                "vlenma $33, $61, $42, $42\n"
                "vlds $61, (7*0x100+0x20*2)(%7)\n"
                "vlenma $46, $58, $59, $59\n"
                "vlenma $33, $58, $41, $41\n"
                "vlds $58, (7*0x100+0x20*3)(%7)\n"
                "vlenma $46, $52, $57, $57\n"
                "vlenma $33, $52, $39, $39\n"
                "vlds $52, (7*0x100+0x20*4)(%7)\n"
                "vlenma $46, $51, $55, $55\n"
                "vlenma $33, $51, $38, $38\n"
                "vlds $51, (7*0x100+0x20*5)(%7)\n"
                "vlenma $46, $50, $54, $54\n"
                "vlenma $33, $50, $37, $37\n"
                "vlds $50, (7*0x100+0x20*6)(%7)\n"
                "vlenma $46, $49, $53, $53\n"
                "vlenma $33, $49, $36, $36\n"
                //#1
                "vlds $49, (7*0x100+0x20*7)(%7)\n"
                "vlenma $46, $45, $47, $47\n"
                "vlenma $33, $45, $44, $44\n"
                "vlds $45,(8*0x100+0x0)(%7)\n"
                "vlenma $46, $62, $63, $63\n"
                "vlenma $33, $62, $43, $43\n"
                "vlds $62,(8*0x100+0x20*1)(%7)\n"
                "vlenma $46, $61, $60, $60\n"
                "vlenma $33, $61, $42, $42\n"
                "vlds $61,(8*0x100+0x20*2)(%7)\n"
                "vlenma $46, $58, $59, $59\n"
                "vlenma $33, $58, $41, $41\n"
                "vlds $58,(8*0x100+0x20*3)(%7)\n"
                "vlenma $46, $52, $57, $57\n"
                "vlenma $33, $52, $39, $39\n"
                "vlds $52,(8*0x100+0x20*4)(%7)\n"
                "vlenma $46, $51, $55, $55\n"
                "vlenma $33, $51, $38, $38\n"
                "vlds $51,(8*0x100+0x20*5)(%7)\n"
                "vlenma $46, $50, $54, $54\n"
                "vlenma $33, $50, $37, $37\n"
                "vlds $50,(8*0x100+0x20*6)(%7)\n"
                "vlenma $46, $49, $53, $53\n"
                "vlenma $33, $49, $36, $36\n"
                "vlds $46, 0x20(%6)\n"
                "vlds $33, 0x120(%6)\n"
                "vlds $49,(8*0x100+0x20*7)(%7)\n"
                : "=&r"(reg), "=&r"(va), "=&r"(vb), "=&r"(vc), "=&r"(vaa),
                  "=&r"(vcc)
                : "r"(&a_ldm[(i << 6) + k]), "r"(&b_ldm[(k << 6) + 0]),
                  "r"(&d_ldm[(i << 6) + 0]));
        }
        // the LAST Loop K
        asm volatile(
            //#0
            "vlenma $46, $45, $47, $47\n"
            "vlenma $33, $45, $44, $44\n"
            "vlds $45, 1*0x100(%7)\n"
            "vlenma $46, $62, $63, $63\n"
            "vlenma $33, $62, $43, $43\n"
            "vlds $62, (1*0x100+0x20*1)(%7)\n"
            "vlenma $46, $61, $60, $60\n"
            "vlenma $33, $61, $42, $42\n"
            "vlds $61, (1*0x100+0x20*2)(%7)\n"
            "vlenma $46, $58, $59, $59\n"
            "vlenma $33, $58, $41, $41\n"
            "vlds $58, (1*0x100+0x20*3)(%7)\n"
            "vlenma $46, $52, $57, $57\n"
            "vlenma $33, $52, $39, $39\n"
            "vlds $52, (1*0x100+0x20*4)(%7)\n"
            "vlenma $46, $51, $55, $55\n"
            "vlenma $33, $51, $38, $38\n"
            "vlds $51, (1*0x100+0x20*5)(%7)\n"
            "vlenma $46, $50, $54, $54\n"
            "vlenma $33, $50, $37, $37\n"
            "vlds $50, (1*0x100+0x20*6)(%7)\n"
            "vlenma $46, $49, $53, $53\n"
            "vlenma $33, $49, $36, $36\n"
            //#1
            "vlds $49, (1*0x100+0x20*7)(%7)\n"
            "vlenma $46, $45, $47, $47\n"
            "vlenma $33, $45, $44, $44\n"
            "vlds $45, 2*0x100(%7)\n"
            "vlenma $46, $62, $63, $63\n"
            "vlenma $33, $62, $43, $43\n"
            "vlds $62, (2*0x100+0x20*1)(%7)\n"
            "vlenma $46, $61, $60, $60\n"
            "vlenma $33, $61, $42, $42\n"
            "vlds $61, (2*0x100+0x20*2)(%7)\n"
            "vlenma $46, $58, $59, $59\n"
            "vlenma $33, $58, $41, $41\n"
            "vlds $58, (2*0x100+0x20*3)(%7)\n"
            "vlenma $46, $52, $57, $57\n"
            "vlenma $33, $52, $39, $39\n"
            "vlds $52, (2*0x100+0x20*4)(%7)\n"
            "vlenma $46, $51, $55, $55\n"
            "vlenma $33, $51, $38, $38\n"
            "vlds $51, (2*0x100+0x20*5)(%7)\n"
            "vlenma $46, $50, $54, $54\n"
            "vlenma $33, $50, $37, $37\n"
            "vlds $50, (2*0x100+0x20*6)(%7)\n"
            "vlenma $46, $49, $53, $53\n"
            "vlenma $33, $49, $36, $36\n"
            //#2
            "vlds $49, (2*0x100+0x20*7)(%7)\n"
            "vlenma $46, $45, $47, $47\n"
            "vlenma $33, $45, $44, $44\n"
            "vlds $45, 3*0x100(%7)\n"
            "vlenma $46, $62, $63, $63\n"
            "vlenma $33, $62, $43, $43\n"
            "vlds $62, (3*0x100+0x20*1)(%7)\n"
            "vlenma $46, $61, $60, $60\n"
            "vlenma $33, $61, $42, $42\n"
            "vlds $61, (3*0x100+0x20*2)(%7)\n"
            "vlenma $46, $58, $59, $59\n"
            "vlenma $33, $58, $41, $41\n"
            "vlds $58, (3*0x100+0x20*3)(%7)\n"
            "vlenma $46, $52, $57, $57\n"
            "vlenma $33, $52, $39, $39\n"
            "vlds $52, (3*0x100+0x20*4)(%7)\n"
            "vlenma $46, $51, $55, $55\n"
            "vlenma $33, $51, $38, $38\n"
            "vlds $51, (3*0x100+0x20*5)(%7)\n"
            "vlenma $46, $50, $54, $54\n"
            "vlenma $33, $50, $37, $37\n"
            "vlds $50, (3*0x100+0x20*6)(%7)\n"
            "vlenma $46, $49, $53, $53\n"
            "vlenma $33, $49, $36, $36\n"
            //#3
            "vlds $49, (3*0x100+0x20*7)(%7)\n"
            "vlenma $46, $45, $47, $47\n"
            "vlenma $33, $45, $44, $44\n"
            "vlds $45, 4*0x100(%7)\n"
            "vlenma $46, $62, $63, $63\n"
            "vlenma $33, $62, $43, $43\n"
            "vlds $62, (4*0x100+0x20*1)(%7)\n"
            "vlenma $46, $61, $60, $60\n"
            "vlenma $33, $61, $42, $42\n"
            "vlds $61, (4*0x100+0x20*2)(%7)\n"
            "vlenma $46, $58, $59, $59\n"
            "vlenma $33, $58, $41, $41\n"
            "vlds $58, (4*0x100+0x20*3)(%7)\n"
            "vlenma $46, $52, $57, $57\n"
            "vlenma $33, $52, $39, $39\n"
            "vlds $52, (4*0x100+0x20*4)(%7)\n"
            "vlenma $46, $51, $55, $55\n"
            "vlenma $33, $51, $38, $38\n"
            "vlds $51, (4*0x100+0x20*5)(%7)\n"
            "vlenma $46, $50, $54, $54\n"
            "vlenma $33, $50, $37, $37\n"
            "vlds $50, (4*0x100+0x20*6)(%7)\n"
            "vlenma $46, $49, $53, $53\n"
            "vlenma $33, $49, $36, $36\n"
            //#4
            "vlds $49, (4*0x100+0x20*7)(%7)\n"
            "vlenma $46, $45, $47, $47\n"
            "vlenma $33, $45, $44, $44\n"
            "vlds $45, 5*0x100(%7)\n"
            "vlenma $46, $62, $63, $63\n"
            "vlenma $33, $62, $43, $43\n"
            "vlds $62, (5*0x100+0x20*1)(%7)\n"
            "vlenma $46, $61, $60, $60\n"
            "vlenma $33, $61, $42, $42\n"
            "vlds $61, (5*0x100+0x20*2)(%7)\n"
            "vlenma $46, $58, $59, $59\n"
            "vlenma $33, $58, $41, $41\n"
            "vlds $58, (5*0x100+0x20*3)(%7)\n"
            "vlenma $46, $52, $57, $57\n"
            "vlenma $33, $52, $39, $39\n"
            "vlds $52, (5*0x100+0x20*4)(%7)\n"
            "vlenma $46, $51, $55, $55\n"
            "vlenma $33, $51, $38, $38\n"
            "vlds $51, (5*0x100+0x20*5)(%7)\n"
            "vlenma $46, $50, $54, $54\n"
            "vlenma $33, $50, $37, $37\n"
            "vlds $50, (5*0x100+0x20*6)(%7)\n"
            "vlenma $46, $49, $53, $53\n"
            "vlenma $33, $49, $36, $36\n"
            //#5
            "vlds $49, (5*0x100+0x20*7)(%7)\n"
            "vlenma $46, $45, $47, $47\n"
            "vlenma $33, $45, $44, $44\n"
            "vlds $45, 6*0x100(%7)\n"
            "vlenma $46, $62, $63, $63\n"
            "vlenma $33, $62, $43, $43\n"
            "vlds $62, (6*0x100+0x20*1)(%7)\n"
            "vlenma $46, $61, $60, $60\n"
            "vlenma $33, $61, $42, $42\n"
            "vlds $61, (6*0x100+0x20*2)(%7)\n"
            "vlenma $46, $58, $59, $59\n"
            "vlenma $33, $58, $41, $41\n"
            "vlds $58, (6*0x100+0x20*3)(%7)\n"
            "vlenma $46, $52, $57, $57\n"
            "vlenma $33, $52, $39, $39\n"
            "vlds $52, (6*0x100+0x20*4)(%7)\n"
            "vlenma $46, $51, $55, $55\n"
            "vlenma $33, $51, $38, $38\n"
            "vlds $51, (6*0x100+0x20*5)(%7)\n"
            "vlenma $46, $50, $54, $54\n"
            "vlenma $33, $50, $37, $37\n"
            "vlds $50, (6*0x100+0x20*6)(%7)\n"
            "vlenma $46, $49, $53, $53\n"
            "vlenma $33, $49, $36, $36\n"
            //#6
            "vlds $49, (6*0x100+0x20*7)(%7)\n"
            "vlenma $46, $45, $47, $47\n"
            "vlenma $33, $45, $44, $44\n"
            "vlds $45, 7*0x100(%7)\n"
            "vlenma $46, $62, $63, $63\n"
            "vlenma $33, $62, $43, $43\n"
            "vlds $62, (7*0x100+0x20*1)(%7)\n"
            "vlenma $46, $61, $60, $60\n"
            "vlenma $33, $61, $42, $42\n"
            "vlds $61, (7*0x100+0x20*2)(%7)\n"
            "vlenma $46, $58, $59, $59\n"
            "vlenma $33, $58, $41, $41\n"
            "vlds $58, (7*0x100+0x20*3)(%7)\n"
            "vlenma $46, $52, $57, $57\n"
            "vlenma $33, $52, $39, $39\n"
            "vlds $52, (7*0x100+0x20*4)(%7)\n"
            "vlenma $46, $51, $55, $55\n"
            "vlenma $33, $51, $38, $38\n"
            "vlds $51, (7*0x100+0x20*5)(%7)\n"
            "vlenma $46, $50, $54, $54\n"
            "vlenma $33, $50, $37, $37\n"
            "vlds $50, (7*0x100+0x20*6)(%7)\n"
            "vlenma $46, $49, $53, $53\n"
            "vlenma $33, $49, $36, $36\n"
            //#1
            "vlds $49, (7*0x100+0x20*7)(%7)\n"
            "vlenma $46, $45, $47, $47\n"
            "vlenma $46, $62, $63, $63\n"
            "vlenma $46, $61, $60, $60\n"
            "vlenma $46, $58, $59, $59\n"
            "vlenma $46, $52, $57, $57\n"
            "vlenma $46, $51, $55, $55\n"
            "vlenma $46, $50, $54, $54\n"
            "vlenma $46, $49, $53, $53\n"
            "vsts   $47, 0x0(%8)\n"
            "vlenma $33, $45, $44, $44\n"
            "vsts   $63, 0x20(%8)\n"
            "vlenma $33, $62, $43, $43\n"
            "vsts   $60, 0x40(%8)\n"
            "vlenma $33, $61, $42, $42\n"
            "vsts   $59, 0x60(%8)\n"
            "vlenma $33, $58, $41, $41\n"
            "vsts   $57, 0x80(%8)\n"
            "vlenma $33, $52, $39, $39\n"
            "vsts   $55, 0xa0(%8)\n"
            "vlenma $33, $51, $38, $38\n"
            "vsts   $54, 0xc0(%8)\n"
            "vlenma $33, $50, $37, $37\n"
            "vsts   $53, 0xe0(%8)\n"
            "vlenma $33, $49, $36, $36\n"
            "vsts   $44, 0x100(%8)\n"
            "vsts   $43, 0x120(%8)\n"
            "vsts   $42, 0x140(%8)\n"
            "vsts   $41, 0x160(%8)\n"
            "vsts   $39, 0x180(%8)\n"
            "vsts   $38, 0x1a0(%8)\n"
            "vsts   $37, 0x1c0(%8)\n"
            "vsts   $36, 0x1e0(%8)\n"
            : "=&r"(reg), "=&r"(va), "=&r"(vb), "=&r"(vc), "=&r"(vaa),
              "=&r"(vcc)
            : "r"(&a_ldm[(i << 6) + k]), "r"(&b_ldm[(k << 6) + 0]),
              "r"(&d_ldm[(i << 6) + 0]));
    }
#endif
}

void kernel_vlenmas_nn_32x32_asm_fp0(float *a_ldm, float *b_ldm, float *d_ldm) {
#if 0
    int i, j, k;
    floatv8 va, vb, vc;
    for(i=0; i<32; i++)
    for(k=0; k<32; k++){
        float A_Part = (-1.0)*a_ldm[i*32+k];
        va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part, A_Part, A_Part, A_Part, A_Part);
        for(j=0; j<32; j+=8){
            simd_load(vc, &d_ldm[i*32+j]);
            simd_load(vb, &b_ldm[k*32+j]);
            vc = simd_vmas(va, vb, vc);
            simd_store(vc, &d_ldm[i*32+j]);
        }
    }
#else
    int i, j, k, reg;
    doublev8 va, vb, vc, vaa, vcc;
    asm volatile("pws");
    asm volatile("ldi  %0, 0x20($31)\n"
                 "wcsr %0, 0x91\n"
                 "ldi  %0, 0x7($31)\n"
                 "wcsr %0, 0x92\n"
                 "ldi  %0, 0x0(%1)\n"
                 "wcsr %0, 0x90\n"
                 : "=&r"(reg)
                 : "r"(d_ldm));
    // asm("pws\n");
    for (i = 0; i < 32; i += 2) {
        asm volatile("vlds $47, 0x0(%8)\n"
                     "vlds $63, 0x20*1(%8)\n"
                     "vlds $60, 0x20*2(%8)\n"
                     "vlds $59, 0x20*3(%8)\n"
                     "vlds $44, (0x80+0x20*0)(%8)\n"
                     "vlds $43, (0x80+0x20*1)(%8)\n"
                     "vlds $42, (0x80+0x20*2)(%8)\n"
                     "vlds $41, (0x80+0x20*3)(%8)\n"
                     "vlds $46, 0x0(%6)\n"
                     "vlds $33, 0x80(%6)\n"
                     "vlds $45,0x0(%7)\n"
                     "vlds $62,0x20*1(%7)\n"
                     "vlds $61,0x20*2(%7)\n"
                     "vlds $58,0x20*3(%7)\n"
                     : "=&r"(reg), "=&r"(va), "=&r"(vb), "=&r"(vc), "=&r"(vaa),
                       "=&r"(vcc)
                     : "r"(&a_ldm[(i << 5) + 0]), "r"(&b_ldm[(0 << 5) + 0]),
                       "r"(&d_ldm[(i << 5) + 0]));
        for (k = 0; k < 24; k += 8) {
            asm volatile(
                //#0
                "vlenma $46, $45, $47, $47\n"
                "vlenma $33, $45, $44, $44\n"
                "vlds $45, 1*0x80(%7)\n"
                "vlenma $46, $62, $63, $63\n"
                "vlenma $33, $62, $43, $43\n"
                "vlds $62, (1*0x80+0x20*1)(%7)\n"
                "vlenma $46, $61, $60, $60\n"
                "vlenma $33, $61, $42, $42\n"
                "vlds $61, (1*0x80+0x20*2)(%7)\n"
                "vlenma $46, $58, $59, $59\n"
                "vlenma $33, $58, $41, $41\n"
                "vlds $58, (1*0x80+0x20*3)(%7)\n"
                //#1
                "vlenma $46, $45, $47, $47\n"
                "vlenma $33, $45, $44, $44\n"
                "vlds $45, 2*0x80(%7)\n"
                "vlenma $46, $62, $63, $63\n"
                "vlenma $33, $62, $43, $43\n"
                "vlds $62, (2*0x80+0x20*1)(%7)\n"
                "vlenma $46, $61, $60, $60\n"
                "vlenma $33, $61, $42, $42\n"
                "vlds $61, (2*0x80+0x20*2)(%7)\n"
                "vlenma $46, $58, $59, $59\n"
                "vlenma $33, $58, $41, $41\n"
                "vlds $58, (2*0x80+0x20*3)(%7)\n"
                //#2
                "vlenma $46, $45, $47, $47\n"
                "vlenma $33, $45, $44, $44\n"
                "vlds $45, 3*0x80(%7)\n"
                "vlenma $46, $62, $63, $63\n"
                "vlenma $33, $62, $43, $43\n"
                "vlds $62, (3*0x80+0x20*1)(%7)\n"
                "vlenma $46, $61, $60, $60\n"
                "vlenma $33, $61, $42, $42\n"
                "vlds $61, (3*0x80+0x20*2)(%7)\n"
                "vlenma $46, $58, $59, $59\n"
                "vlenma $33, $58, $41, $41\n"
                "vlds $58, (3*0x80+0x20*3)(%7)\n"
                //#3
                "vlenma $46, $45, $47, $47\n"
                "vlenma $33, $45, $44, $44\n"
                "vlds $45, 4*0x80(%7)\n"
                "vlenma $46, $62, $63, $63\n"
                "vlenma $33, $62, $43, $43\n"
                "vlds $62, (4*0x80+0x20*1)(%7)\n"
                "vlenma $46, $61, $60, $60\n"
                "vlenma $33, $61, $42, $42\n"
                "vlds $61, (4*0x80+0x20*2)(%7)\n"
                "vlenma $46, $58, $59, $59\n"
                "vlenma $33, $58, $41, $41\n"
                "vlds $58, (4*0x80+0x20*3)(%7)\n"
                //#4
                "vlenma $46, $45, $47, $47\n"
                "vlenma $33, $45, $44, $44\n"
                "vlds $45, 5*0x80(%7)\n"
                "vlenma $46, $62, $63, $63\n"
                "vlenma $33, $62, $43, $43\n"
                "vlds $62, (5*0x80+0x20*1)(%7)\n"
                "vlenma $46, $61, $60, $60\n"
                "vlenma $33, $61, $42, $42\n"
                "vlds $61, (5*0x80+0x20*2)(%7)\n"
                "vlenma $46, $58, $59, $59\n"
                "vlenma $33, $58, $41, $41\n"
                "vlds $58, (5*0x80+0x20*3)(%7)\n"
                //#5
                "vlenma $46, $45, $47, $47\n"
                "vlenma $33, $45, $44, $44\n"
                "vlds $45, 6*0x80(%7)\n"
                "vlenma $46, $62, $63, $63\n"
                "vlenma $33, $62, $43, $43\n"
                "vlds $62, (6*0x80+0x20*1)(%7)\n"
                "vlenma $46, $61, $60, $60\n"
                "vlenma $33, $61, $42, $42\n"
                "vlds $61, (6*0x80+0x20*2)(%7)\n"
                "vlenma $46, $58, $59, $59\n"
                "vlenma $33, $58, $41, $41\n"
                "vlds $58, (6*0x80+0x20*3)(%7)\n"
                //#6
                "vlenma $46, $45, $47, $47\n"
                "vlenma $33, $45, $44, $44\n"
                "vlds $45, 7*0x80(%7)\n"
                "vlenma $46, $62, $63, $63\n"
                "vlenma $33, $62, $43, $43\n"
                "vlds $62, (7*0x80+0x20*1)(%7)\n"
                "vlenma $46, $61, $60, $60\n"
                "vlenma $33, $61, $42, $42\n"
                "vlds $61, (7*0x80+0x20*2)(%7)\n"
                "vlenma $46, $58, $59, $59\n"
                "vlenma $33, $58, $41, $41\n"
                "vlds $58, (7*0x80+0x20*3)(%7)\n"
                //#7
                "vlenma $46, $45, $47, $47\n"
                "vlenma $33, $45, $44, $44\n"
                "vlds $45,(8*0x80+0x0)(%7)\n"
                "vlenma $46, $62, $63, $63\n"
                "vlenma $33, $62, $43, $43\n"
                "vlds $62,(8*0x80+0x20*1)(%7)\n"
                "vlenma $46, $61, $60, $60\n"
                "vlenma $33, $61, $42, $42\n"
                "vlds $61,(8*0x80+0x20*2)(%7)\n"
                "vlenma $46, $58, $59, $59\n"
                "vlenma $33, $58, $41, $41\n"
                "vlds $58,(8*0x80+0x20*3)(%7)\n"
                "vlds $46, 0x20(%6)\n"
                "vlds $33, (0x80+0x20)(%6)\n"
                : "=&r"(reg), "=&r"(va), "=&r"(vb), "=&r"(vc), "=&r"(vaa),
                  "=&r"(vcc)
                : "r"(&a_ldm[(i << 5) + k]), "r"(&b_ldm[(k << 5) + 0]),
                  "r"(&d_ldm[(i << 5) + 0]));
        }
        // the LAST Loop K
        asm volatile(
            //#0
            "vlenma $46, $45, $47, $47\n"
            "vlenma $33, $45, $44, $44\n"
            "vlds $45, 1*0x80(%7)\n"
            "vlenma $46, $62, $63, $63\n"
            "vlenma $33, $62, $43, $43\n"
            "vlds $62, (1*0x80+0x20*1)(%7)\n"
            "vlenma $46, $61, $60, $60\n"
            "vlenma $33, $61, $42, $42\n"
            "vlds $61, (1*0x80+0x20*2)(%7)\n"
            "vlenma $46, $58, $59, $59\n"
            "vlenma $33, $58, $41, $41\n"
            "vlds $58, (1*0x80+0x20*3)(%7)\n"
            //#1
            "vlenma $46, $45, $47, $47\n"
            "vlenma $33, $45, $44, $44\n"
            "vlds $45, 2*0x80(%7)\n"
            "vlenma $46, $62, $63, $63\n"
            "vlenma $33, $62, $43, $43\n"
            "vlds $62, (2*0x80+0x20*1)(%7)\n"
            "vlenma $46, $61, $60, $60\n"
            "vlenma $33, $61, $42, $42\n"
            "vlds $61, (2*0x80+0x20*2)(%7)\n"
            "vlenma $46, $58, $59, $59\n"
            "vlenma $33, $58, $41, $41\n"
            "vlds $58, (2*0x80+0x20*3)(%7)\n"
            //#2
            "vlenma $46, $45, $47, $47\n"
            "vlenma $33, $45, $44, $44\n"
            "vlds $45, 3*0x80(%7)\n"
            "vlenma $46, $62, $63, $63\n"
            "vlenma $33, $62, $43, $43\n"
            "vlds $62, (3*0x80+0x20*1)(%7)\n"
            "vlenma $46, $61, $60, $60\n"
            "vlenma $33, $61, $42, $42\n"
            "vlds $61, (3*0x80+0x20*2)(%7)\n"
            "vlenma $46, $58, $59, $59\n"
            "vlenma $33, $58, $41, $41\n"
            "vlds $58, (3*0x80+0x20*3)(%7)\n"
            //#3
            "vlenma $46, $45, $47, $47\n"
            "vlenma $33, $45, $44, $44\n"
            "vlds $45, 4*0x80(%7)\n"
            "vlenma $46, $62, $63, $63\n"
            "vlenma $33, $62, $43, $43\n"
            "vlds $62, (4*0x80+0x20*1)(%7)\n"
            "vlenma $46, $61, $60, $60\n"
            "vlenma $33, $61, $42, $42\n"
            "vlds $61, (4*0x80+0x20*2)(%7)\n"
            "vlenma $46, $58, $59, $59\n"
            "vlenma $33, $58, $41, $41\n"
            "vlds $58, (4*0x80+0x20*3)(%7)\n"
            //#4
            "vlenma $46, $45, $47, $47\n"
            "vlenma $33, $45, $44, $44\n"
            "vlds $45, 5*0x80(%7)\n"
            "vlenma $46, $62, $63, $63\n"
            "vlenma $33, $62, $43, $43\n"
            "vlds $62, (5*0x80+0x20*1)(%7)\n"
            "vlenma $46, $61, $60, $60\n"
            "vlenma $33, $61, $42, $42\n"
            "vlds $61, (5*0x80+0x20*2)(%7)\n"
            "vlenma $46, $58, $59, $59\n"
            "vlenma $33, $58, $41, $41\n"
            "vlds $58, (5*0x80+0x20*3)(%7)\n"
            //#5
            "vlenma $46, $45, $47, $47\n"
            "vlenma $33, $45, $44, $44\n"
            "vlds $45, 6*0x80(%7)\n"
            "vlenma $46, $62, $63, $63\n"
            "vlenma $33, $62, $43, $43\n"
            "vlds $62, (6*0x80+0x20*1)(%7)\n"
            "vlenma $46, $61, $60, $60\n"
            "vlenma $33, $61, $42, $42\n"
            "vlds $61, (6*0x80+0x20*2)(%7)\n"
            "vlenma $46, $58, $59, $59\n"
            "vlenma $33, $58, $41, $41\n"
            "vlds $58, (6*0x80+0x20*3)(%7)\n"
            //#6
            "vlenma $46, $45, $47, $47\n"
            "vlenma $33, $45, $44, $44\n"
            "vlds $45, 7*0x80(%7)\n"
            "vlenma $46, $62, $63, $63\n"
            "vlenma $33, $62, $43, $43\n"
            "vlds $62, (7*0x80+0x20*1)(%7)\n"
            "vlenma $46, $61, $60, $60\n"
            "vlenma $33, $61, $42, $42\n"
            "vlds $61, (7*0x80+0x20*2)(%7)\n"
            "vlenma $46, $58, $59, $59\n"
            "vlenma $33, $58, $41, $41\n"
            "vlds $58, (7*0x80+0x20*3)(%7)\n"
            //#1
            "vlenma $46, $45, $47, $47\n"
            "vlenma $46, $62, $63, $63\n"
            "vlenma $46, $61, $60, $60\n"
            "vlenma $46, $58, $59, $59\n"
            "vsts   $47, 0x0(%8)\n"
            "vlenma $33, $45, $44, $44\n"
            "vsts   $63, 0x20(%8)\n"
            "vlenma $33, $62, $43, $43\n"
            "vsts   $60, 0x40(%8)\n"
            "vlenma $33, $61, $42, $42\n"
            "vsts   $59, 0x60(%8)\n"
            "vlenma $33, $58, $41, $41\n"
            "vsts   $44, 0x80(%8)\n"
            "vsts   $43, 0xa0(%8)\n"
            "vsts   $42, 0xc0(%8)\n"
            "vsts   $41, 0xe0(%8)\n"
            : "=&r"(reg), "=&r"(va), "=&r"(vb), "=&r"(vc), "=&r"(vaa),
              "=&r"(vcc)
            : "r"(&a_ldm[(i << 5) + k]), "r"(&b_ldm[(k << 5) + 0]),
              "r"(&d_ldm[(i << 5) + 0]));
    }
#endif
}

void kernel_vlenmas_nn_64x64_asm_simd(float *a_ldm, float *b_ldm,
                                      float *d_ldm) {
    float ALPHA = 1.0;
#if 0
    int i, j, k;
    floatv8 va, vb, vc;
    for(i=0; i<64; i++)
    for(k=0; k<64; k++){
        float A_Part = ALPHA*a_ldm[i*64+k];
        va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part, A_Part, A_Part, A_Part, A_Part);
        for(j=0; j<64; j+=8){
            simd_load(vc, &d_ldm[i*64+j]);
            simd_load(vb, &b_ldm[k*64+j]);
            vc = simd_vmas(va, vb, vc);
            simd_store(vc, &d_ldm[i*64+j]);
        }
    }
#else
    int i, j, k;
    register floatv8 va0, va1, vb0, vb1, vb2, vb3, vb4, vb5, vb6, vb7, vc00,
        vc01, vc02, vc03, vc04, vc05, vc06, vc07, vc10, vc11, vc12, vc13, vc14,
        vc15, vc16, vc17;
    for (i = 0; i < 64; i += 2) {
        for (k = 0; k < 64; k++) {
            register float A_Part0 = ALPHA * a_ldm[(i << 6) + k];
            register float A_Part1 = ALPHA * a_ldm[((i + 1) << 6) + k];
            va0 = simd_set_floatv8(A_Part0, A_Part0, A_Part0, A_Part0, A_Part0,
                                   A_Part0, A_Part0, A_Part0);
            va1 = simd_set_floatv8(A_Part1, A_Part1, A_Part1, A_Part1, A_Part1,
                                   A_Part1, A_Part1, A_Part1);

            simd_load(vb0, &b_ldm[k << 6]);
            simd_load(vc00, &d_ldm[i << 6]);
            simd_load(vc10, &d_ldm[(i + 1) << 6]);

            simd_load(vb1, &b_ldm[(k << 6) + 8]);
            simd_load(vc01, &d_ldm[(i << 6) + 8]);
            simd_load(vc11, &d_ldm[((i + 1) << 6) + 8]);

            vc00 = simd_vmas(va0, vb0, vc00);
            vc01 = simd_vmas(va0, vb1, vc01);
            vc10 = simd_vmas(va1, vb0, vc10);
            vc11 = simd_vmas(va1, vb1, vc11);

            simd_load(vb2, &b_ldm[(k << 6) + 16]);
            simd_load(vc02, &d_ldm[(i << 6) + 16]);
            simd_load(vc12, &d_ldm[((i + 1) << 6) + 16]);

            simd_load(vb3, &b_ldm[(k << 6) + 24]);
            simd_load(vc03, &d_ldm[(i << 6) + 24]);
            simd_load(vc13, &d_ldm[((i + 1) << 6) + 24]);

            vc02 = simd_vmas(va0, vb2, vc02);
            vc03 = simd_vmas(va0, vb3, vc03);
            vc12 = simd_vmas(va1, vb2, vc12);
            vc13 = simd_vmas(va1, vb3, vc13);

            simd_load(vb4, &b_ldm[(k << 6) + 32]);
            simd_load(vc04, &d_ldm[(i << 6) + 32]);
            simd_load(vc14, &d_ldm[((i + 1) << 6) + 32]);

            simd_load(vb5, &b_ldm[(k << 6) + 40]);
            simd_load(vc05, &d_ldm[(i << 6) + 40]);
            simd_load(vc15, &d_ldm[((i + 1) << 6) + 40]);

            vc04 = simd_vmas(va0, vb4, vc04);
            vc05 = simd_vmas(va0, vb5, vc05);
            vc14 = simd_vmas(va1, vb4, vc14);
            vc15 = simd_vmas(va1, vb5, vc15);

            simd_load(vb6, &b_ldm[(k << 6) + 48]);
            simd_load(vc06, &d_ldm[(i << 6) + 48]);
            simd_load(vc16, &d_ldm[((i + 1) << 6) + 48]);

            simd_load(vb7, &b_ldm[(k << 6) + 56]);
            simd_load(vc07, &d_ldm[(i << 6) + 56]);
            simd_load(vc17, &d_ldm[((i + 1) << 6) + 56]);

            vc06 = simd_vmas(va0, vb6, vc06);
            vc07 = simd_vmas(va0, vb7, vc07);
            vc16 = simd_vmas(va1, vb6, vc16);
            vc17 = simd_vmas(va1, vb7, vc17);

            simd_store(vc00, &d_ldm[i << 6]);
            simd_store(vc01, &d_ldm[(i << 6) + 8]);
            simd_store(vc02, &d_ldm[(i << 6) + 16]);
            simd_store(vc03, &d_ldm[(i << 6) + 24]);
            simd_store(vc04, &d_ldm[(i << 6) + 32]);
            simd_store(vc05, &d_ldm[(i << 6) + 40]);
            simd_store(vc06, &d_ldm[(i << 6) + 48]);
            simd_store(vc07, &d_ldm[(i << 6) + 56]);

            simd_store(vc10, &d_ldm[(i + 1) << 6]);
            simd_store(vc11, &d_ldm[((i + 1) << 6) + 8]);
            simd_store(vc12, &d_ldm[((i + 1) << 6) + 16]);
            simd_store(vc13, &d_ldm[((i + 1) << 6) + 24]);
            simd_store(vc14, &d_ldm[((i + 1) << 6) + 32]);
            simd_store(vc15, &d_ldm[((i + 1) << 6) + 40]);
            simd_store(vc16, &d_ldm[((i + 1) << 6) + 48]);
            simd_store(vc17, &d_ldm[((i + 1) << 6) + 56]);
        }
    }
#endif
}

void kernel_vlenmas_nn_32x32_asm_simd(float *a32_ldm, float *b32_ldm,
                                      float *c32_ldm) {
    float ALPHA = 1.0;
#if 0
    int i, j, k;
	floatv8 va, vb, vc;
	for (i = 0; i < 32; i++) {
	    for (k = 0; k < 32; k++) {
		float A_Part = ALPHA * a32_ldm[i * 32 + k];
		va = simd_set_floatv8(A_Part, A_Part, A_Part, A_Part, A_Part, A_Part, A_Part, A_Part);
		for (j = 0; j < 32; j += 8) {
			simd_load(vc, &c32_ldm[i * 32 + j]);
			simd_load(vb, &b32_ldm[k * 32 + j]);
			vc = simd_vmas(va, vb, vc);
			simd_store(vc, &c32_ldm[i * 32 + j]);
		}
            }
	}
#else
    int i, j, k;
    register floatv8 va0, va1, va2, va3, vb0, vb1, vb2, vb3, vc00, vc01, vc02,
        vc03, vc20, vc21, vc22, vc23, vc10, vc11, vc12, vc13, vc30, vc31, vc32,
        vc33;
    for (i = 0; i < 32; i += 4) {
        for (k = 0; k < 32; k++) {
            register float A_Part0 = ALPHA * a32_ldm[(i << 5) + k];
            register float A_Part1 = ALPHA * a32_ldm[((i + 1) << 5) + k];
            register float A_Part2 = ALPHA * a32_ldm[((i + 2) << 5) + k];
            register float A_Part3 = ALPHA * a32_ldm[((i + 3) << 5) + k];
            va0 = simd_set_floatv8(A_Part0, A_Part0, A_Part0, A_Part0, A_Part0,
                                   A_Part0, A_Part0, A_Part0);
            va1 = simd_set_floatv8(A_Part1, A_Part1, A_Part1, A_Part1, A_Part1,
                                   A_Part1, A_Part1, A_Part1);
            va2 = simd_set_floatv8(A_Part2, A_Part2, A_Part2, A_Part2, A_Part2,
                                   A_Part2, A_Part2, A_Part2);
            va3 = simd_set_floatv8(A_Part3, A_Part3, A_Part3, A_Part3, A_Part3,
                                   A_Part3, A_Part3, A_Part3);

            simd_load(vb0, &b32_ldm[k << 5]);
            simd_load(vc00, &c32_ldm[i << 5]);
            simd_load(vc10, &c32_ldm[(i + 1) << 5]);
            simd_load(vc20, &c32_ldm[(i + 2) << 5]);
            simd_load(vc30, &c32_ldm[(i + 3) << 5]);
            simd_load(vb1, &b32_ldm[(k << 5) + 8]);
            simd_load(vc01, &c32_ldm[(i << 5) + 8]);
            simd_load(vc11, &c32_ldm[((i + 1) << 5) + 8]);
            simd_load(vc21, &c32_ldm[((i + 2) << 5) + 8]);
            simd_load(vc31, &c32_ldm[((i + 3) << 5) + 8]);
            simd_load(vb2, &b32_ldm[(k << 5) + 16]);
            simd_load(vc02, &c32_ldm[(i << 5) + 16]);
            simd_load(vc12, &c32_ldm[((i + 1) << 5) + 16]);
            simd_load(vc22, &c32_ldm[((i + 2) << 5) + 16]);
            simd_load(vc32, &c32_ldm[((i + 3) << 5) + 16]);
            simd_load(vb3, &b32_ldm[(k << 5) + 24]);
            simd_load(vc03, &c32_ldm[(i << 5) + 24]);
            simd_load(vc13, &c32_ldm[((i + 1) << 5) + 24]);
            simd_load(vc23, &c32_ldm[((i + 2) << 5) + 24]);
            simd_load(vc33, &c32_ldm[((i + 3) << 5) + 24]);

            vc00 = simd_vmas(va0, vb0, vc00);
            vc01 = simd_vmas(va0, vb1, vc01);
            vc10 = simd_vmas(va1, vb0, vc10);
            vc11 = simd_vmas(va1, vb1, vc11);
            vc20 = simd_vmas(va2, vb0, vc20);
            vc21 = simd_vmas(va2, vb1, vc21);
            vc30 = simd_vmas(va3, vb0, vc30);
            vc31 = simd_vmas(va3, vb1, vc31);
            vc02 = simd_vmas(va0, vb2, vc02);
            vc03 = simd_vmas(va0, vb3, vc03);
            vc12 = simd_vmas(va1, vb2, vc12);
            vc13 = simd_vmas(va1, vb3, vc13);
            vc22 = simd_vmas(va2, vb2, vc22);
            vc23 = simd_vmas(va2, vb3, vc23);
            vc32 = simd_vmas(va3, vb2, vc32);
            vc33 = simd_vmas(va3, vb3, vc33);

            simd_store(vc00, &c32_ldm[i << 5]);
            simd_store(vc01, &c32_ldm[(i << 5) + 8]);
            simd_store(vc10, &c32_ldm[(i + 1) << 5]);
            simd_store(vc11, &c32_ldm[((i + 1) << 5) + 8]);
            simd_store(vc20, &c32_ldm[(i + 2) << 5]);
            simd_store(vc21, &c32_ldm[((i + 2) << 5) + 8]);
            simd_store(vc30, &c32_ldm[(i + 3) << 5]);
            simd_store(vc31, &c32_ldm[((i + 3) << 5) + 8]);
            simd_store(vc02, &c32_ldm[(i << 5) + 16]);
            simd_store(vc03, &c32_ldm[(i << 5) + 24]);
            simd_store(vc12, &c32_ldm[((i + 1) << 5) + 16]);
            simd_store(vc13, &c32_ldm[((i + 1) << 5) + 24]);
            simd_store(vc22, &c32_ldm[((i + 2) << 5) + 16]);
            simd_store(vc23, &c32_ldm[((i + 2) << 5) + 24]);
            simd_store(vc32, &c32_ldm[((i + 3) << 5) + 16]);
            simd_store(vc33, &c32_ldm[((i + 3) << 5) + 24]);
        }
    }
#endif
}
