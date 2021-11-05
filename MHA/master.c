/*************************************************************************
	> File Name: convolution_forward.c
	> Author: 
	> Mail: 
	> Created Time: 
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <athread.h>

#include "args.h"
#include "util.h"
#include "common.h"

extern void SLAVE_FUN(gemm_nt_block)();
extern void SLAVE_FUN(gemm_nt_block_nn)();
extern void SLAVE_FUN(gemm_tn_block)();
extern void SLAVE_FUN(gemm_nt_block_qkv)();
extern void SLAVE_FUN(gemm_nn_block)();
extern void SLAVE_FUN(dma_memset)();
extern void SLAVE_FUN(dma_trans_head)();
extern void SLAVE_FUN(dma_norm)();
extern void SLAVE_FUN(dma_norm2)();
#define YI(b,s,d) (((b)*S+s)*D+d)
#define NI(b,n,s,pd) ((((b)*N+n)*S+s)*PD+pd)
#define QKI(b,n,sh,sl) ((((b)*N+n)*S+sh)*S+sl)
#define KVI(b,n,pdh,pdl) ((((b)*N+n)*PD+pdh)*PD+pdl)

void get_xxc(int *Mc,int *Nc, int *Kc, int M, int N){
    int Ldmc=32;
    int mc=Ldmc,nc=Ldmc,kc=Ldmc;
    while(mc>4){
        if(mc*8<=M)break;
        mc/=2;
    }
    while(nc>4){
        if(nc*8<=N)break;
        nc/=2;
    }
    int Mmax=mc;
    if(nc>mc)Mmax=nc;
    kc=Ldmc*Ldmc/Mmax;
    *Mc=mc;
    *Nc=nc;
    *Kc=kc;
}

int multihead_attention(Args_t arg)
{
	const int B = arg->B;
    const int S = arg->S;
    const int D = arg->D;
    const int N = arg->N;
    const float* x = arg->x;
    const float* w = arg->w;
    float* Q = arg->Q;
    float* K = arg->K;
    float* V = arg->V;
    float* QK = arg->QK;
    float* y = arg->y;
	const int PD = D/N;
	float* QN = (float*)aligned_malloc(sizeof(float)*B*N*S*PD, 128);
	float* KN = (float*)aligned_malloc(sizeof(float)*B*N*S*PD, 128);
	float* VN = (float*)aligned_malloc(sizeof(float)*B*N*S*PD, 128);
	float* KV = (float*)aligned_malloc(sizeof(float)*B*N*PD*PD, 128);
    Re_Args_t reArg=(Re_Args_t)malloc(sizeof(reArg));
  
    reArg->A=x;
    reArg->B=w;
    reArg->QN=Q;
    reArg->KN=K;
    reArg->VN=V;
    reArg->M=B*S;
    reArg->N=D;
    reArg->K=D;
    reArg->D=D;
    reArg->LDA=D;
    reArg->LDB=D;
    reArg->LDC=D;
    get_xxc(&reArg->Mc,&reArg->Nc,&reArg->Kc,reArg->M,reArg->N);
    athread_spawn(gemm_nt_block_qkv,reArg);
    athread_join();

    reArg->A=Q;
    reArg->B=K;
    reArg->C=V;
    reArg->QN=QN;
    reArg->VN=VN;
    reArg->KN=KN;
    reArg->M=B;
    reArg->N=S;
    reArg->D=D;
    reArg->K=N;
    athread_spawn(dma_trans_head,reArg);
    athread_join();
	
	if(S<(PD<<1)){
        reArg->LDA=PD;
        reArg->LDB=PD;
        reArg->LDC=S;
        reArg->M=S;
        reArg->N=S;
        reArg->K=PD;
        get_xxc(&reArg->Mc,&reArg->Nc,&reArg->Kc,reArg->M,reArg->N);
        for(int b = 0; b < B; b ++)
            for(int n = 0; n < N; n ++){
                reArg->A=QN+NI(b,n,0,0);
                reArg->B=KN+NI(b,n,0,0);
                reArg->C=QK+QKI(b,n,0,0);
                athread_spawn(gemm_nt_block, reArg); // spawn
                athread_join(); // wait for all slave threads finished
            }

        reArg->buf=QK;
        reArg->M=B*N*S;
        reArg->N=S;
        athread_spawn(dma_norm, reArg);
        athread_join();

        reArg->M=S;
        reArg->N=PD;
        reArg->K=S;
        reArg->LDA=S;
        reArg->LDB=PD;
        reArg->LDC=D;    
        get_xxc(&reArg->Mc,&reArg->Nc,&reArg->Kc,reArg->M,reArg->N);
        for (int b = 0; b < B; b++) {
            for (int n = 0; n < N; n++) {
                reArg->A=QK+QKI(b,n,0,0);
                reArg->B=VN+NI(b,n,0,0);
                reArg->C=y+b*S*D+n*PD;
                athread_spawn(gemm_nn_block,reArg);
                athread_join();
            }
        }
    }else{
        reArg->LDA=PD;
        reArg->LDB=PD;
        reArg->LDC=S;
        reArg->M=S;
        reArg->N=S;
        reArg->K=PD;
        get_xxc(&reArg->Mc,&reArg->Nc,&reArg->Kc,reArg->M,reArg->N);
        for(int b = 0; b < B; b ++)
            for(int n = 0; n < N; n ++){
                reArg->A=QN+NI(b,n,0,0);
                reArg->B=KN+NI(b,n,0,0);
                reArg->C=QK+QKI(b,n,0,0);
                athread_spawn(gemm_nt_block_nn, reArg); // spawn
                athread_join(); // wait for all slave threads finished
            }

        reArg->LDA=PD;
        reArg->LDB=PD;
        reArg->LDC=PD;
        reArg->M=PD;
        reArg->N=PD;
        reArg->K=S;
        get_xxc(&reArg->Mc,&reArg->Nc,&reArg->Kc,reArg->M,reArg->N);
        for(int b = 0; b < B; b ++)
            for(int n = 0; n < N; n ++){
                reArg->A=KN+NI(b,n,0,0);
                reArg->B=VN+NI(b,n,0,0);
                reArg->C=KV+KVI(b,n,0,0);
                athread_spawn(gemm_tn_block, reArg); // spawn
                athread_join();
            }

        reArg->M=S;
        reArg->N=PD;
        reArg->K=PD;
        reArg->LDA=PD;
        reArg->LDB=PD;
        reArg->LDC=D;
        get_xxc(&reArg->Mc,&reArg->Nc,&reArg->Kc,reArg->M,reArg->N);
        for (int b = 0; b < B; b++) {
            for (int n = 0; n < N; n++) {
                reArg->A=QN+NI(b,n,0,0);
                reArg->B=KV+KVI(b,n,0,0);
                reArg->C=y+b*S*D+n*PD;
                athread_spawn(gemm_nn_block,reArg);
                athread_join();
            }
        }

        reArg->M=S;
        reArg->N=S;
        reArg->K=PD;
        reArg->D=D;
        for(int b = 0; b < B; b ++)
            for(int n = 0; n < N; n ++){
                reArg->buf=QK+QKI(b,n,0,0);
                reArg->buf2=y+YI(b,0,n*PD);
                athread_spawn(dma_norm2, reArg);
                athread_join();
            }
    }
    
	aligned_free(QN);
	aligned_free(KN);
	aligned_free(VN);
	aligned_free(KV);
    free(reArg);
    return 0;
}

