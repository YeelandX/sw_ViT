#include "args.h"
#ifndef _COMMON_H
#define _COMMON_H
typedef struct ReArgs
{
	int LDA;
    int LDB;
    int LDC;
    int M;
    int N;
    int K;
    int D;
    float* A;
    float* B;
    float* C;
    float* QN;
    float* KN;
    float* VN;
    float* buf;
    float* buf2;
    int Mc;
    int Nc;
    int Kc;
}ReArgs, *Re_Args_t;

#endif
