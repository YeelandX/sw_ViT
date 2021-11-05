#include <slave.h>
#include <math.h>
#include <simd.h>
#include <assert.h>
#include <string.h>
#include "args.h"
#include "common.h"
#include "dma.h"

#define _ali __attribute__((__aligned__(128)))
#define rpcc(time) asm volatile("rcsr %0,4" :  "=r" (time));
#define GEMM_LDM 32*32+128
#define NORM_LDM 1024+128

__thread_local volatile unsigned long get_reply, put_reply,reply;
__thread_local float a_ldm[GEMM_LDM] _ali;
__thread_local float b_ldm[GEMM_LDM] _ali, b_ldm1[GEMM_LDM] _ali, b_ldm2[GEMM_LDM] _ali;
__thread_local float bb_ldm[GEMM_LDM] _ali, bb_ldm1[GEMM_LDM] _ali, bb_ldm2[GEMM_LDM] _ali;
__thread_local float c_ldm[GEMM_LDM] _ali, c_ldm1[GEMM_LDM] _ali, c_ldm2[GEMM_LDM] _ali;
__thread_local float buf_ldm[NORM_LDM] _ali, buf2_ldm[NORM_LDM] _ali;
__thread_local static float buf3_ldm[NORM_LDM] _ali;
__thread_local float src_ldm[128] _ali, src_ldm1[128] _ali, src_ldm2[128] _ali;

#define DMA_SET(d,size,bsize,stepsize,reply) {\
	  dma_set_size(d, size); \
	  dma_set_bsize(d, bsize); \
	  dma_set_stepsize(d, stepsize); \
	  dma_set_reply(d, reply); \
	  dma_set_mode(d, PE_MODE); \
}

#define DMA_SET_NOSTEP(d,mode,size,reply) {\
	  dma_set_size(d, size); \
	  dma_set_op(d, mode); \
	  dma_set_mode(d, PE_MODE); \
	  dma_set_reply(d, reply); \
}


void gemm_tn_block(Re_Args_t reArg){
	const int M = reArg->M;		
	const int N = reArg->N;
	const int K = reArg->K;
	const int LDA = reArg->LDA;
	const int LDB = reArg->LDB;
	const int LDC = reArg->LDC;
	const float* A = reArg->A;
	const float* B = reArg->B;
	float* C = reArg->C;
	const int Mc=reArg->Mc;
	const int Nc=reArg->Nc;
	const int Kc=reArg->Kc;
	int i,j,k,ii,jj,kk,pp,i_end,j_end,k_end;
	float A_Part;
	floatv4 va,vb,vc;

	// ----------dma init begin----------
	dma_desc get_desc=0;
	dma_desc put_desc=0;
	dma_set_op(&get_desc, DMA_GET);
	dma_set_op(&put_desc, DMA_PUT);
	// ----------dma init end----------

	i_end=Mc;
	for(ii=_ROW*Mc; ii<M; ii+=Mc<<3){
		if(ii+Mc>M)i_end=M-ii;
		j_end=Nc;
		for(jj=_COL*Nc; jj<N; jj+=Nc<<3){
			if(jj+Nc>N) j_end=N-jj;
			// ----------memset begin----------
			for(pp=0;pp<i_end*j_end;pp++)c_ldm[pp]=0;
			// ----------memset end----------
			k_end=Kc;
			for(kk=0; kk<K; kk+=Kc){
				if(kk+Kc>K)k_end=K-kk;
				// ----------dma get A&B begin----------
				get_reply=0;
				DMA_SET(&get_desc,(i_end*k_end)<<2,i_end<<2,(LDA-i_end)<<2,&get_reply);
				dma(get_desc,&A[kk*LDA+ii],a_ldm);
				DMA_SET(&get_desc,(j_end*k_end)<<2,j_end<<2,(LDB-j_end)<<2,&get_reply);
				dma(get_desc,&B[kk*LDB+jj],b_ldm);
				dma_wait(&get_reply,2);
				// ----------dma get A&B end----------

				// ----------calculate kernal begin----------
				for(k = 0; k < k_end; k++){
					for(i = 0; i < i_end; i++){
						A_Part=a_ldm[k*i_end+i];
						va=simd_set_floatv4(A_Part,A_Part,A_Part,A_Part);
						for(j = 0; j < j_end; j+=4){
							simd_load(vc,&c_ldm[i*j_end+j]);
							simd_load(vb,&b_ldm[k*j_end+j]);
							vc=simd_vmas(va,vb,vc);
							simd_store(vc,&c_ldm[i*j_end+j]);
						}
					}
				}
				// ----------calculate kernal end----------
			}
			
			// ----------dma put C begin----------
			put_reply=0;
			DMA_SET(&put_desc,(i_end*j_end)<<2,j_end<<2,(LDC-j_end)<<2,&put_reply);
			dma(put_desc,&C[ii*LDC+jj],c_ldm);
			dma_wait(&put_reply,1);
			// ----------dma put C end----------
		}
	}
}


void gemm_nt_block_nn(Re_Args_t reArg){
	const int M = reArg->M;
	const int N = reArg->N;
	const int K = reArg->K;
	const int LDA = reArg->LDA;
	const int LDB = reArg->LDB;
	const int LDC = reArg->LDC;
	const float* A = reArg->A;
	const float* B = reArg->B;
	float* C = reArg->C;
	const int Mc=reArg->Mc;
	const int Nc=reArg->Nc;
	const int Kc=reArg->Kc;
	float res[4];
	int i,j,k,ii,jj,kk,pp,i_end=Mc,j_end=Nc,k_end=Kc,i_k,j_k,i_j,k_j;
	floatv4 va,vb,vc;
	floatv4 va0, vb00, vb01, vb02, vb03, vc00, vc01, vc02, vc03, vc10, vc11, vc12, vc13;
	float *addr_a, *addr_b, *addr_c, A_Part;

	// ----------dma init begin----------
	dma_desc get_desc=0;
	dma_desc put_desc=0;
	dma_set_op(&get_desc, DMA_GET);
	dma_set_op(&put_desc, DMA_PUT);
	// ----------dma init end----------

	i_end=Mc;
	for(ii=_ROW*Mc; ii<M; ii+=Mc<<3){
		if(ii+Mc>M)i_end=M-ii;
		j_end=Nc;
		addr_a=&A[ii*LDA];
		addr_c=&C[ii*LDC];
		for(jj=_COL*Nc; jj<N; jj+=Nc<<3){
			if(jj+Nc>N)j_end=N-jj;
			k_end=Kc;
			addr_b=&B[jj*LDB];
			// ----------memset begin----------
			for(pp=0;pp<i_end*j_end;pp++)c_ldm[pp]=0;
			// ----------memset end----------
			for(kk=0; kk<K; kk+=Kc){
				if(kk+Kc>K) k_end=K-kk;
				// ----------dma get A&B begin----------
				get_reply=0;
				DMA_SET(&get_desc,(i_end*k_end)<<2,k_end<<2,(LDA-k_end)<<2,&get_reply);
				dma(get_desc,addr_a+kk,a_ldm);
				DMA_SET(&get_desc,(j_end*k_end)<<2,k_end<<2,(LDB-k_end)<<2,&get_reply);
				dma(get_desc,addr_b+kk,b_ldm);
				dma_wait(&get_reply,2);
				// ----------dma get A&B end----------
				
				//-------transpose of matrix begin------
				for(j=0;j<j_end;j++){
					for(k=0;k<k_end;k++){
						bb_ldm[k*j_end+j]=b_ldm[j*k_end+k];
					}
				}
				//-------transpose of matrix end------

				// ----------calculate kernal begin(nn)----------
				for(i = 0; i < i_end; i++){
					i_j = i * j_end;
					i_k = i * k_end;
					for(k = 0; k < k_end; k++){
						k_j = k * j_end;
						register float A_Part0 = a_ldm[i_k + k];
						va0 = simd_set_floatv4(A_Part0, A_Part0, A_Part0, A_Part0);
						if(j_end==20){
							simd_load(vb00, &bb_ldm[k_j]);
							simd_load(vb01, &bb_ldm[k_j+4]);
							simd_load(vb02, &bb_ldm[k_j+8]);
							simd_load(vb03, &bb_ldm[k_j+12]);
							simd_load(vc00, &c_ldm[i_j]);
							simd_load(vc01, &c_ldm[i_j+4]);
							simd_load(vc02, &c_ldm[i_j+8]);
							simd_load(vc03, &c_ldm[i_j+12]);
							vc00 = simd_vmas(va0, vb00, vc00);
							vc01 = simd_vmas(va0, vb01, vc01);
							vc02 = simd_vmas(va0, vb02, vc02);
							simd_load(vb00, &bb_ldm[k_j+16]);
							simd_load(vc10, &c_ldm[i_j+16]);
							vc03 = simd_vmas(va0, vb03, vc03);
							vc10 = simd_vmas(va0, vb00, vc10);
							simd_store(vc00, &c_ldm[i_j]);
							simd_store(vc01, &c_ldm[i_j+4]);
							simd_store(vc02, &c_ldm[i_j+8]);
							simd_store(vc03, &c_ldm[i_j+12]);
							simd_store(vc10, &c_ldm[i_j+16]);
						}else{
							simd_load(vb00, &bb_ldm[k_j]);
							simd_load(vb01, &bb_ldm[k_j+4]);
							simd_load(vb02, &bb_ldm[k_j+8]);
							simd_load(vb03, &bb_ldm[k_j+12]);
							simd_load(vc00, &c_ldm[i_j]);
							simd_load(vc01, &c_ldm[i_j+4]);
							simd_load(vc02, &c_ldm[i_j+8]);
							simd_load(vc03, &c_ldm[i_j+12]);
							vc00 = simd_vmas(va0, vb00, vc00);
							vc01 = simd_vmas(va0, vb01, vc01);
							vc02 = simd_vmas(va0, vb02, vc02);
							vc03 = simd_vmas(va0, vb03, vc03);
							simd_load(vb00, &bb_ldm[k_j+16]);
							simd_load(vb01, &bb_ldm[k_j+20]);
							simd_load(vb02, &bb_ldm[k_j+24]);
							simd_load(vb03, &bb_ldm[k_j+28]);
							simd_load(vc10, &c_ldm[i_j+16]);
							simd_load(vc11, &c_ldm[i_j+20]);
							simd_load(vc12, &c_ldm[i_j+24]);
							simd_load(vc13, &c_ldm[i_j+28]);
							vc10 = simd_vmas(va0, vb00, vc10);
							vc11 = simd_vmas(va0, vb01, vc11);
							vc12 = simd_vmas(va0, vb02, vc12);
							vc13 = simd_vmas(va0, vb03, vc13);
							simd_store(vc00, &c_ldm[i_j]);
							simd_store(vc01, &c_ldm[i_j+4]);
							simd_store(vc02, &c_ldm[i_j+8]);
							simd_store(vc03, &c_ldm[i_j+12]);
							simd_store(vc10, &c_ldm[i_j+16]);
							simd_store(vc11, &c_ldm[i_j+20]);
							simd_store(vc12, &c_ldm[i_j+24]);
							simd_store(vc13, &c_ldm[i_j+28]);
						}
					}
				}
				// ----------calculate kernal end----------
			}

			// ----------dma put C begin----------
			put_reply=0;
			DMA_SET(&put_desc,(i_end*j_end)<<2,j_end<<2,(LDC-j_end)<<2,&put_reply);
			dma(put_desc,addr_c+jj,c_ldm);
			dma_wait(&put_reply,1);
			// ----------dma put C end----------
		}
	}
}

void gemm_nt_block(Re_Args_t reArg){
	const int M = reArg->M;
	const int N = reArg->N;
	const int K = reArg->K;
	const int LDA = reArg->LDA;
	const int LDB = reArg->LDB;
	const int LDC = reArg->LDC;
	const float* A = reArg->A;
	const float* B = reArg->B;
	float* C = reArg->C;
	const int Mc=reArg->Mc;
	const int Nc=reArg->Nc;
	const int Kc=reArg->Kc;
	float res[4];
	int i,j,k,ii,jj,kk,pp,i_end=Mc,j_end=Nc,k_end=Kc,i_k,j_k;
	floatv4 va,vb,vc;
	float *addr_a, *addr_b, *addr_c;

	// ----------dma init begin----------
	dma_desc get_desc=0;
	dma_desc put_desc=0;
	dma_set_op(&get_desc, DMA_GET);
	dma_set_op(&put_desc, DMA_PUT);
	// ----------dma init end----------

	i_end=Mc;
	for(ii=_ROW*Mc; ii<M; ii+=Mc<<3){
		if(ii+Mc>M)i_end=M-ii;
		addr_a=&A[ii*LDA];
		addr_c=&C[ii*LDC];
		j_end=Nc;
		for(jj=_COL*Nc; jj<N; jj+=Nc<<3){
			if(jj+Nc>N)j_end=N-jj;
			addr_b=&B[jj*LDB];
			// ----------memset begin----------
			for(pp=0;pp<i_end*j_end;pp++)c_ldm[pp]=0;
			// ----------memset end----------
			k_end=Kc;
			for(kk=0; kk<K; kk+=Kc){
				if(kk+Kc>K) k_end=K-kk;
				// ----------dma get A&B begin----------
				get_reply=0;
				DMA_SET(&get_desc,(i_end*k_end)<<2,k_end<<2,(LDA-k_end)<<2,&get_reply);
				dma(get_desc,addr_a+kk,a_ldm);
				DMA_SET(&get_desc,(j_end*k_end)<<2,k_end<<2,(LDB-k_end)<<2,&get_reply);
				dma(get_desc,addr_b+kk,b_ldm);
				dma_wait(&get_reply,2);
				// ----------dma get A&B end----------

				// ----------calculate kernal begin----------
				for(i = 0; i < i_end; i++){
					i_k=i*k_end;
					for(j = 0; j < j_end; j++){
						vc=0;
						j_k=j*k_end;
						for(k = 0; k < k_end; k+=4){
							simd_load(va,&a_ldm[i_k+k]);
							simd_load(vb,&b_ldm[j_k+k]);
							vc=simd_vmas(va,vb,vc);
						}
						simd_store(vc, res);
						c_ldm[i*j_end+j]+=res[0]+res[1]+res[2]+res[3];
					}
				}
				// ----------calculate kernal end----------
			}

			// ----------dma put C begin----------
			put_reply=0;
			DMA_SET(&put_desc,(i_end*j_end)<<2,j_end<<2,(LDC-j_end)<<2,&put_reply);
			dma(put_desc,addr_c+jj,c_ldm);
			dma_wait(&put_reply,1);
			// ----------dma put C end----------
		}
	}
}

void gemm_nt_block_qkv(Re_Args_t reArg){
	const int M = reArg->M;
	const int N = reArg->N;
	const int K = reArg->K;
	const int LDA = reArg->LDA;
	const int LDB = reArg->LDB;
	const int LDC = reArg->LDC;
	const int D=reArg->D;
	const float* A = reArg->A;
	const float* B = reArg->B;
	const float* B1 = reArg->B+D*D;
	const float* B2 = reArg->B+2*D*D;
	float* C = reArg->QN;
	float* C1 = reArg->KN;
	float* C2 = reArg->VN;
	const int Mc=reArg->Mc;
	const int Nc=reArg->Nc;
	const int Kc=reArg->Kc;
	float res[4],A_Part;
	int i,j,k,ii,jj,kk,pp,i_end=Mc,j_end=Nc,k_end=Kc,i_k,j_k,i_j;
	floatv4 va,vb,vc,vb1,vc1,vb2,vc2;

	// ----------dma init begin----------
	dma_desc get_desc=0;
	dma_desc put_desc=0;
	dma_set_op(&get_desc, DMA_GET);
	dma_set_op(&put_desc, DMA_PUT);
	// ----------dma init end----------

	i_end=Mc;
	for(ii=_ROW*Mc; ii<M; ii+=Mc<<3){
		if(ii+Mc>M)i_end=M-ii;
		for(jj=_COL*Nc; jj<N; jj+=Nc<<3){
			// ----------memset begin----------
			for(pp=0;pp<(i_end*j_end);pp++){
				c_ldm[pp]=0;
				c_ldm1[pp]=0;
				c_ldm2[pp]=0;
			}
			// ----------memset end----------
			for(kk=0; kk<K; kk+=Kc){
				// ----------dma get A&B begin----------
				get_reply=0;
				DMA_SET(&get_desc,(i_end*k_end)<<2,k_end<<2,(LDA-k_end)<<2,&get_reply);
				dma(get_desc,&A[ii*LDA+kk],a_ldm);
				DMA_SET(&get_desc,(j_end*k_end)<<2,k_end<<2,(LDB-k_end)<<2,&get_reply);
				dma(get_desc,&B[jj*LDB+kk],b_ldm);
				dma(get_desc,&B1[jj*LDB+kk],b_ldm1);
				dma(get_desc,&B2[jj*LDB+kk],b_ldm2);
				dma_wait(&get_reply,4);
				// ----------dma get A&B end----------

				// ----------calculate kernal begin----------
				for(i = 0; i < i_end; i++){
					i_k=i*k_end;
					for(j = 0; j < j_end; j++){
						vc=0;vc1=0;vc2=0;
						j_k=j*k_end;
						for(k = 0; k < k_end; k+=4){
							simd_load(va,&a_ldm[i_k+k]);
							simd_load(vb,&b_ldm[j_k+k]);
							vc=simd_vmas(va,vb,vc);
							simd_load(vb1,&b_ldm1[j_k+k]);
							vc1=simd_vmas(va,vb1,vc1);
							simd_load(vb2,&b_ldm2[j_k+k]);
							vc2=simd_vmas(va,vb2,vc2);
						}
						simd_store(vc, res);
						c_ldm[i*j_end+j]+=res[0]+res[1]+res[2]+res[3];
						simd_store(vc1, res);
						c_ldm1[i*j_end+j]+=res[0]+res[1]+res[2]+res[3];
						simd_store(vc2, res);
						c_ldm2[i*j_end+j]+=res[0]+res[1]+res[2]+res[3];
					}
				}
				// ----------calculate kernal end----------
			}
			// ----------dma put C begin----------
			put_reply=0;
			DMA_SET(&put_desc,(i_end*j_end)<<2,j_end<<2,(LDC-j_end)<<2,&put_reply);
			dma(put_desc,&C[ii*LDC+jj],c_ldm);
			dma(put_desc,&C1[ii*LDC+jj],c_ldm1);
			dma(put_desc,&C2[ii*LDC+jj],c_ldm2);
			dma_wait(&put_reply,3);
			// ----------dma put C end----------
		}
	}
}

void gemm_nn_block(Re_Args_t reArg){
	int M = reArg->M;
	int N = reArg->N;
	int K = reArg->K;
	const int LDA = reArg->LDA;
	const int LDB = reArg->LDB;
	const int LDC = reArg->LDC;
	const float* A = reArg->A;
	const float* B = reArg->B;
	float* C = reArg->C;
	float A_Part;
	floatv4 va0,va1,vb00,vb01,vb02,vb03,vb10,vb11,vb12,vb13,vc00,vc01,vc02,vc03,vc10,vc11,vc12,vc13;
	int Mc=reArg->Mc, Nc=reArg->Nc,Kc=reArg->Kc;
	int i,j,k,ii,jj,kk,pp,i_end=Mc,j_end=Nc,k_end=Kc,i_j,k_j,i_k;
	float *addr_a, *addr_c;

	// ----------dma init begin----------
	dma_desc get_desc=0;
	dma_desc put_desc=0;
	dma_set_op(&get_desc, DMA_GET);
	dma_set_op(&put_desc, DMA_PUT);
	// ----------dma init begin----------

	for(ii=_ROW*Mc; ii<M; ii+=Mc<<3){
		if(ii+Mc>M)i_end=M-ii;
		j_end=Nc;
		addr_a=&A[ii*LDA];
		addr_c=&C[ii*LDC];
		for(jj=_COL*Nc; jj<N; jj+=Nc<<3){
			if(jj+Nc>N) j_end=N-jj;
			// ----------memset begin----------
			for(pp=0;pp<i_end*j_end;pp++)c_ldm[pp]=0;
			// ----------memset end----------
			k_end=Kc;
			for(kk=0; kk<K; kk+=Kc){
				if(kk+Kc>K)k_end=K-kk;
				// ----------dma get A&B begin----------
				get_reply=0;
				DMA_SET(&get_desc,(i_end*k_end)<<2,k_end<<2,(LDA-k_end)<<2,&get_reply);
				dma(get_desc,addr_a+kk,a_ldm);
				DMA_SET(&get_desc,(j_end*k_end)<<2,j_end<<2,(LDB-j_end)<<2,&get_reply);
				dma(get_desc,&B[kk*LDB+jj],b_ldm);
				dma_wait(&get_reply,2);
				// ----------dma get A&B end----------
				
				// ----------calculate kernal begin----------
				for(i = 0; i < i_end; i++){
					i_j = i * j_end;
					i_k = i * k_end;
					for(k = 0; k < k_end; k++){
						k_j = k * j_end;
						register float A_Part0 = a_ldm[i_k + k];
						va0 = simd_set_floatv4(A_Part0, A_Part0, A_Part0, A_Part0);
						if (j_end == 8) {
							simd_load(vb00, &b_ldm[k_j]);//vb
							simd_load(vb01, &b_ldm[k_j + 4]);
							simd_load(vc00, &c_ldm[i_j]);//vc
							simd_load(vc01, &c_ldm[i_j + 4]);
							vc00 = simd_vmas(va0, vb00, vc00);
							vc01 = simd_vmas(va0, vb01, vc01);
							simd_store(vc00, &c_ldm[i_j]);
							simd_store(vc01, &c_ldm[i_j + 4]);
						} else {
							simd_load(vb00, &b_ldm[k_j]);
							simd_load(vb01, &b_ldm[k_j + 4]);
							simd_load(vb02, &b_ldm[k_j + 8]);
							simd_load(vb03, &b_ldm[k_j + 12]);
							simd_load(vc00, &c_ldm[i_j]);
							simd_load(vc01, &c_ldm[i_j + 4]);
							simd_load(vc02, &c_ldm[i_j + 8]);
							simd_load(vc03, &c_ldm[i_j + 12]);
							vc00 = simd_vmas(va0, vb00, vc00);
							vc01 = simd_vmas(va0, vb01, vc01);
							vc02 = simd_vmas(va0, vb02, vc02);
							vc03 = simd_vmas(va0, vb03, vc03);
							simd_store(vc00, &c_ldm[i_j]);
							simd_store(vc01, &c_ldm[i_j + 4]);
							simd_store(vc02, &c_ldm[i_j + 8]);
							simd_store(vc03, &c_ldm[i_j + 12]);
						}
					}
				}
				// ----------calculate kernal end----------
			}
			// ----------dma put C begin----------
			put_reply=0;
			DMA_SET(&put_desc,(i_end*j_end)<<2,j_end<<2,(LDC-j_end)<<2,&put_reply);
			dma(put_desc,addr_c+jj,c_ldm);
			dma_wait(&put_reply,1);
			// ----------dma put C end----------
		}
	}
}

void dma_norm(Re_Args_t reArg){
	float* BUF = reArg->buf;
	const int N=reArg->N;
	const int M=reArg->M;
	int i,j,k;
	double sum;

	dma_desc get_desc=0;
	dma_desc put_desc=0;
	DMA_SET_NOSTEP(&get_desc,DMA_GET,N<<2,&get_reply);
	DMA_SET_NOSTEP(&put_desc,DMA_PUT,N<<2,&put_reply);

	for(i=_MYID;i<M;i+=64){
		sum = 0.0f;
		get_reply=0;
		dma_set_reply(&get_desc,&get_reply);
		dma(get_desc,&BUF[i*N], buf_ldm);
		dma_wait(&get_reply, 1);
		for(int i = 0;i < N; i ++)
			sum += buf_ldm[i];
		for(int i = 0;i < N;i ++)
			buf_ldm[i] /= sum;
		put_reply=0;
		dma_set_reply(&put_desc,&put_reply);
		dma(put_desc,&BUF[i*N],&buf_ldm[0]);
		dma_wait(&put_reply,1);
	}
}

void dma_norm2(Re_Args_t reArg){
	float* BUF = reArg->buf;
	float* BUF2 = reArg->buf2;
	const int N=reArg->N;
	const int M=reArg->M;
	const int K=reArg->K;
	const int D=reArg->D;
	int i,j,k;
	double sum;

	dma_desc get_desc=0;
	dma_desc put_desc=0;
	dma_set_op(&get_desc,DMA_GET);
	dma_set_op(&put_desc, DMA_PUT);
	dma_set_size(&put_desc, K<<2);
	for(i=_MYID;i<M;i+=64){
		sum = 0.0f;
		get_reply=0;
		dma_set_reply(&get_desc,&get_reply);
		dma_set_size(&get_desc, N<<2);
		dma(get_desc,&BUF[i*N], buf_ldm);
		dma_set_size(&get_desc, K<<2);
		dma(get_desc,&BUF2[i*D], buf2_ldm);
		dma_wait(&get_reply, 2);
		for(int i = 0; i < N; i++)
			sum += buf_ldm[i];
		for(int i = 0; i < K; i++)
			buf2_ldm[i] /= sum;
		put_reply=0;
		dma_set_reply(&put_desc,&put_reply);
		dma(put_desc,&BUF2[i*D],buf2_ldm);
		dma_wait(&put_reply,1);
	}
}

void dma_trans_head(Re_Args_t reArg){
	int B=reArg->M;
	int S=reArg->N;
	int D=reArg->D;
	int N=reArg->K;
	int PD=D/N;
	float* srcQ = reArg->A;
	float* srcK = reArg->B;
	float* srcV = reArg->C;
	float* dstQ = reArg->QN;
	float* dstK = reArg->KN;
	float* dstV = reArg->VN;
	dma_desc get_desc=0;
	dma_desc put_desc=0;
	dma_set_op(&get_desc, DMA_GET);
	dma_set_op(&put_desc, DMA_PUT);
	dma_set_size(&get_desc,PD<<2);
	dma_set_size(&put_desc,PD<<2);
	int sd=S*D,nspd=N*S*PD,spd=S*PD;

	for(int b=0;b<B;b++){
		for(int n=0;n<N;n++){
			for(int s=_MYID;s<S;s+=64){
				get_reply=0;
				dma_set_reply(&get_desc,&get_reply);
				dma(get_desc,&srcQ[b*sd+s*D+n*PD],src_ldm);
				dma(get_desc,&srcK[b*sd+s*D+n*PD],src_ldm1);
				dma(get_desc,&srcV[b*sd+s*D+n*PD],src_ldm2);
				dma_wait(&get_reply,3);

				put_reply=0;
				dma_set_reply(&put_desc,&put_reply);
				dma(put_desc,&dstQ[b*nspd+s*PD+n*spd],src_ldm);
				dma(put_desc,&dstK[b*nspd+s*PD+n*spd],src_ldm1);
				dma(put_desc,&dstV[b*nspd+s*PD+n*spd],src_ldm2);
				dma_wait(&put_reply,3);
			}
		}
	}
}
