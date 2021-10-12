#ifdef __cplusplus
extern "C" {
#endif
int swptex_mm(const void *A, const void *B, void *C, size_t M, size_t N,
              size_t K, int transposeA, int transposeB);
int swptex_bmm(const void *A, const void *B, void *C, size_t batch, size_t M,
               size_t N, size_t K, int transposeA, int transposeB);
int swptex_dsoftmax(void *dy_, const void *y_, size_t M, size_t N);
int swptex_softmax(void *x_, size_t M, size_t N);
int swptex_transpose_and_merge(void *QKV_, void *Q_, void *K_, void *V_,
                               size_t B, size_t N, size_t S, size_t D);
int swptex_split_and_transpose(void *QKV_, void *Q_, void *K_, void *V_,
                               size_t B, size_t N, size_t S, size_t D);
int swptex_split(const void *QKV_, void *QKVT_, size_t B, size_t N, size_t S,
                 size_t D);
int swptex_merge(const void *QKV_, void *QKVT_, size_t B, size_t N, size_t S,
                 size_t D);
int swptex_scale(void *x_, size_t len, float scaling);
int sum_axis0(const void *src_, void *dst_, size_t M, size_t N);
int swptex_mul(void *x_, size_t len, float ALPHA);
int swptex_relu(void *src_,void *dst_, size_t len);
int swptex_relu_backward(void *x_, void *src_,void *dst_, int len);
int swptex_add(void *x_, void *y_, int len, float ALPHA);
int swptex_addcmul(void *x_, void *t1_, void *t2_, int len, float value);
int swptex_addcdiv(void *x_, void *t1_, void *t2_, int len, float value);


#ifdef __cplusplus
}
#endif
