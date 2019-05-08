#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

REGISTER_OP("Double")
  .Input("to_double: int32")
  .Output("doubled: int32")
  .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);
