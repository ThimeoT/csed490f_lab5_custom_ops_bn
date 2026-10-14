#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>

// python module initialization
// - it registers _C module to Python Interpreter
// - can be used as from. import_C
extern "C" {
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",
          NULL,
          -1,
          NULL,
      };
      return PyModule_Create(&module_def);
  }
}

namespace custom_ops_bn {

// Declarations of CUDA functions
std::vector<at::Tensor> batchnorm_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    at::Tensor& running_mean,
    at::Tensor& running_var,
    bool training,
    double momentum,
    double eps);

    
std::vector<at::Tensor> batchnorm_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& gamma,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd);

// Operator registration
TORCH_LIBRARY(custom_ops_bn, m) {
    m.def("batchnorm_forward(Tensor input, Tensor gamma, Tensor beta, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps) -> Tensor[]");
    m.def("batchnorm_backward(Tensor grad_output, Tensor input, Tensor gamma, Tensor save_mean, Tensor save_invstd) -> Tensor[]");
}

// backend impl for operators (CUDA)
TORCH_LIBRARY_IMPL(custom_ops_bn, CUDA, m) {
    m.impl("batchnorm_forward", &batchnorm_forward_cuda);
    m.impl("batchnorm_backward", &batchnorm_backward_cuda);
}

} // namespace custom_ops_bn
