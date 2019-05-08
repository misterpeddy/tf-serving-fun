#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class DoubleOp : public OpKernel {
  public:
    explicit DoubleOp(OpKernelConstruction *context): OpKernel(context) {}

    // Note this needs to be thread-safe w.r.t class member accesses
    void Compute(OpKernelContext *context) override {
      // Grab input tensor
      const Tensor& input_tensor = context->input(0);
      auto input = input_tensor.flat<int32>();

      // Create output tensor
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                       &output_tensor));
      auto output_flat = output_tensor->flat<int32>();

      // Double each entry
      const int N = input.size();
      for (int i = 0; i < N; i++) {
        output_flat(i) = input(i) * 2;
      }
    }
};

REGISTER_KERNEL_BUILDER(Name("Double").Device(DEVICE_CPU), DoubleOp);
