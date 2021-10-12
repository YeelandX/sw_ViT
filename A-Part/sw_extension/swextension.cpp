
#include <torch/extension.h>

#include "athread.h"
#include "swops.h"
#include <vector>

void swmul(torch::Tensor &self, float scaling) {
  auto len = self.numel();
  auto x = self.data_ptr();
  swptex_mul(x, len, scaling);
}

void swadd(torch::Tensor &self, torch::Tensor &other, float ALPHA) {
  auto len = self.numel();
  auto x = self.data_ptr();
  auto y = other.data_ptr();
  swptex_add(x, y, len, ALPHA);
}

void swaddcmul(torch::Tensor &self, torch::Tensor &tensor1,
               torch::Tensor &tensor2, float value) {
  auto len = self.numel();
  auto x = self.data_ptr();
  auto t1 = tensor1.data_ptr();
  auto t2 = tensor2.data_ptr();
  swptex_addcmul(x, t1, t2, len, value);
}

void swaddcdiv(torch::Tensor &self, torch::Tensor &tensor1,
               torch::Tensor &tensor2, float value) {
  auto len = self.numel();
  auto x = self.data_ptr();
  auto t1 = tensor1.data_ptr();
  auto t2 = tensor2.data_ptr();
  swptex_addcdiv(x, t1, t2, len, value);
}

torch::Tensor swrelu_forward(torch::Tensor input) {
  auto output = torch::zeros_like(input);
  auto len = input.numel();
  auto x = input.data_ptr();
  auto y = output.data_ptr();

  swptex_relu(x, y, len);
  return output;
}

torch::Tensor swrelu_backward(torch::Tensor input, torch::Tensor grad_output) {
  auto output = torch::zeros_like(grad_output);
  auto len = input.numel();
  auto x = input.data_ptr();
  auto dst = output.data_ptr();
  auto src = grad_output.data_ptr();
  swptex_relu_backward(x, src, dst, len);
  return output;
}

std::vector<torch::Tensor> swlinear_forward(torch::Tensor input,
                                            torch::Tensor weight) {
  auto idims = input.sizes();
  auto ndim = idims.size();
  std::vector<int64_t> odims(ndim);

  auto N = weight.size(0);     // output features
  auto K = weight.size(1);     // input features
  auto M = input.numel() / K;  // batch

  for (size_t i = 0; i < ndim - 1; i++) {
    odims[i] = idims[i];
  }
  odims[ndim - 1] = N;

  // Y = X*WT
  auto output = torch::empty(odims);

  // get raw data pointer
  auto x = input.data_ptr();
  auto w = weight.data_ptr();
  auto y = output.data_ptr();

  // call op kernel impl
  swptex_mm(x, w, y, M, N, K, 0, 1);

  return {output};
}

std::vector<torch::Tensor> swlinear_backward(torch::Tensor grad_output,
                                             torch::Tensor input,
                                             torch::Tensor weight,
                                             torch::Tensor bias) {
  auto N = weight.size(0);     // output features
  auto K = weight.size(1);     // input features
  auto M = input.numel() / K;  // batch

  auto d_input = torch::zeros_like(input);
  auto d_weight = torch::zeros_like(weight);
  auto d_bias = torch::zeros_like(bias);

  auto dx = d_input.data_ptr();
  auto dw = d_weight.data_ptr();
  auto db = d_bias.data_ptr();
  auto dy = grad_output.data_ptr();
  auto x = input.data_ptr();
  auto w = weight.data_ptr();

  // dw = dyT * X
  swptex_mm(dy, x, dw, N, K, M, 1, 0);
  // dx = dy * W
  swptex_mm(dy, w, dx, M, K, N, 0, 0);
  // db = sum(dy, axis = 0)
  sum_axis0(dy, db, M, N);

  return {d_input, d_weight, d_bias};
}

std::vector<torch::Tensor> swmha_forward(torch::Tensor input /* (B, S, F) */,
                                         torch::Tensor weight /* (N*D*3, F)
                                         */, int64_t nheads, float scaling) {
  auto B = input.size(0);
  auto S = input.size(1);
  auto F = input.size(2);

  auto N = nheads;
  auto D = weight.size(0) / 3 / N;

  auto qkv = torch::empty({B, S, 3 * N * D});  // WT * X
  auto q = torch::empty({B, N, S, D});         // WqT * Xq
  auto k = torch::empty({B, N, S, D});         // WkT * Xk
  auto v = torch::empty({B, N, S, D});         // WvT * Xv
  auto qk = torch::empty({B, N, S, S});        // softmax(bmm(q, k))
  auto out = torch::empty({B, N, S, D});       // bmm(qk, v)
  auto output = torch::empty({B, S, N * D});

  auto x = input.data_ptr();
  auto w = weight.data_ptr();
  auto y = output.data_ptr();
  auto yT = out.data_ptr();
  auto qkv_data = qkv.data_ptr();
  auto qk_data = qk.data_ptr();
  auto q_data = q.data_ptr();
  auto k_data = k.data_ptr();
  auto v_data = v.data_ptr();

  // unsigned long st, ed;
  swptex_mm(x, w, qkv_data, B * S, 3 * N * D, F, 0, 1);
  swptex_split_and_transpose(qkv_data, q_data, k_data, v_data, B, N, S, D);
  swptex_bmm(q_data, k_data, qk_data, B * N, S, S, D, 0,
             1);  // (S, D) * (S, D)T = (S, S)
  swptex_scale(qk_data, B * N * S * S, scaling);
  swptex_softmax(qk_data, B * N * S, S);
  swptex_bmm(qk_data, v_data, yT, B * N, S, D, S, 0,
             0);                    // (S, S) * (S, D) = (S, D)
  swptex_merge(yT, y, B, N, S, D);  // (B, N, S, D) -> (B, S, (N, D))

  return {output, qk, q, k, v};
}

std::vector<torch::Tensor> swmha_backward(torch::Tensor grad_output,
                                          torch::Tensor input /* (B, S, F)
                                          */, torch::Tensor weight /* (N*D*3,
                                          F) */, torch::Tensor qk,
                                          torch::Tensor q, torch::Tensor k,
                                          torch::Tensor v, float scaling) {
  auto F = input.size(2);
  auto B = q.size(0);
  auto N = q.size(1);
  auto S = q.size(2);
  auto D = q.size(3);

  auto dinput = torch::zeros_like(input);
  auto dweight = torch::zeros_like(weight);
  auto dqkv = torch::zeros({B, S, 3 * N * D});
  auto dout = torch::zeros({B, N, S, D});
  auto dqk = torch::zeros_like(qk);
  auto dq = torch::zeros_like(q);
  auto dk = torch::zeros_like(k);
  auto dv = torch::zeros_like(v);

  auto x = input.data_ptr();
  auto w = weight.data_ptr();
  auto qk_data = qk.data_ptr();
  auto q_data = q.data_ptr();
  auto k_data = k.data_ptr();
  auto v_data = v.data_ptr();

  auto dx = dinput.data_ptr();
  auto dw = dweight.data_ptr();
  auto dyT = dout.data_ptr();
  auto dy = grad_output.data_ptr();
  auto dqkv_data = dqkv.data_ptr();
  auto dqk_data = dqk.data_ptr();
  auto dq_data = dq.data_ptr();
  auto dk_data = dk.data_ptr();
  auto dv_data = dv.data_ptr();

  swptex_split(dy, dyT, B, N, S, D);  // (B, S, (N, D)) -> (B, N, S, D)
  swptex_bmm(qk_data, dyT, dv_data, B * N, S, D, S, 1,
             0);  // (S, S)T * (S, D) = (S, D)
  swptex_bmm(dyT, v_data, dqk_data, B * N, S, S, D, 0,
             1);  // (S, D) * (S, D)T = (S, S)
  swptex_dsoftmax(dqk_data, qk_data, B * N * S, S);
  swptex_scale(dqk_data, B * N * S * S, scaling);  // (S, D) * (S, D)T = (S,S) 
  swptex_bmm(dqk_data, q_data, dk_data, B * N, S, D, S, 1,
             0);  // (S, S)T * (S, D) = (S, D)
  swptex_bmm(dqk_data, k_data, dq_data, B * N, S, D, S, 0,
             0);  // (S, S) * (S, D) = (S, D)
  swptex_transpose_and_merge(dqkv_data, dq_data, dk_data, dv_data, B, N, S,
  D); swptex_mm(dqkv_data, x, dw, 3 * N * D, F, B * S, 1,
            0);  // (B*S, 3*N*D)T * (B*S, F) = (3*N*D, F)
  swptex_mm(dqkv_data, w, dx, B * S, F, 3 * N * D, 0,
            0);  // (B*S, 3*N*D) * (3*N*D, F) = (B*S, F)

  return {dinput, dweight};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swaddcdiv", &swaddcdiv, "swaddcdiv");
  m.def("swaddcmul", &swaddcmul, "swaddcmul");
  m.def("swadd", &swadd, "swadd");
  m.def("swrelu_forward", &swrelu_forward, "swrelu forward");
  m.def("swrelu_backward", &swrelu_backward, "swrelu backward");
  m.def("swmul", &swmul, "swmul");
  m.def("swlinear_forward", &swlinear_forward, "swLinear forward");
  m.def("swlinear_backward", &swlinear_backward, "swLinear backward");
  m.def("swmha_forward", &swmha_forward, "swMha forward");
  m.def("swmha_backward", &swmha_backward, "swMha backward");
}
