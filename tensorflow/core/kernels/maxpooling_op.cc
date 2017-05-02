/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/maxpooling_op.h"

#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/eigen_pooling.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/pooling_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/maxpooling_op_gpu.h"
#include "tensorflow/core/kernels/pooling_ops_common_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

const int kInvalidMaxPoolingIndex = -1;

template <typename Device, typename T>
static void SpatialMaxPoolWithArgMaxHelper(
    OpKernelContext* context, Tensor* output, Tensor* output_arg_max,
    Tensor* input_backprop, const Tensor& tensor_in, const Tensor& out_backprop,
    const PoolParameters& params, const Padding& padding) {
  typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      ConstEigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      EigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<int64, Eigen::Dynamic, Eigen::Dynamic>>
      EigenIndexMatrixMap;

  ConstEigenMatrixMap in_mat(
      tensor_in.flat<T>().data(), params.depth,
      params.tensor_in_cols * params.tensor_in_rows * params.tensor_in_batch);
  EigenMatrixMap out_mat(
      output->flat<T>().data(), params.depth,
      params.out_width * params.out_height * params.tensor_in_batch);
  EigenIndexMatrixMap out_arg_max_mat(
      output_arg_max->flat<int64>().data(), params.depth,
      params.out_width * params.out_height * params.tensor_in_batch);

  const DeviceBase::CpuWorkerThreads& worker_threads =
      *(context->device()->tensorflow_cpu_worker_threads());

  // The following code basically does the following:
  // 1. Flattens the input and output tensors into two dimensional arrays.
  //    tensor_in_as_matrix:
  //      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
  //    output_as_matrix:
  //      depth by (out_width * out_height * tensor_in_batch)
  //
  // 2. Walks through the set of columns in the flattened tensor_in_as_matrix,
  //    and updates the corresponding column(s) in output_as_matrix with the
  //    max value.
  auto shard = [&params, &in_mat, &out_mat, &out_arg_max_mat, &input_backprop,
                &output_arg_max, &out_backprop](int64 start, int64 limit) {

    const int32 depth = params.depth;
    const int32 in_rows = params.tensor_in_rows;
    const int32 in_cols = params.tensor_in_cols;
    const int32 pad_rows = params.pad_rows;
    const int32 pad_cols = params.pad_cols;
    const int32 window_rows = params.window_rows;
    const int32 window_cols = params.window_cols;
    const int32 row_stride = params.row_stride;
    const int32 col_stride = params.col_stride;
    const int32 out_height = params.out_height;
    const int32 out_width = params.out_width;

    {
      // Initializes the output tensor with MIN<T>.
      const int32 output_image_size = out_height * out_width * depth;
      EigenMatrixMap out_shard(out_mat.data() + start * output_image_size, 1,
                               (limit - start) * output_image_size);
      out_shard.setConstant(Eigen::NumTraits<T>::lowest());
      EigenIndexMatrixMap out_arg_max_shard(
          out_arg_max_mat.data() + start * output_image_size, 1,
          (limit - start) * output_image_size);
      out_arg_max_shard.setConstant(kInvalidMaxPoolingIndex);
    }

    for (int64 b = start; b < limit; ++b) {
      for (int h = 0; h < in_rows; ++h) {
        for (int w = 0; w < in_cols; ++w) {
          // (h_start, h_end) * (w_start, w_end) is the range that the input
          // vector projects to.
          const int hpad = h + pad_rows;
          const int wpad = w + pad_cols;
          const int h_start =
              (hpad < window_rows) ? 0 : (hpad - window_rows) / row_stride + 1;
          const int h_end = std::min(hpad / row_stride + 1, out_height);
          const int w_start =
              (wpad < window_cols) ? 0 : (wpad - window_cols) / col_stride + 1;
          const int w_end = std::min(wpad / col_stride + 1, out_width);
          // compute elementwise max
          const int64 in_index = (b * in_rows + h) * in_cols + w;
          for (int ph = h_start; ph < h_end; ++ph) {
            const int64 out_index_base = (b * out_height + ph) * out_width;
            for (int pw = w_start; pw < w_end; ++pw) {
              const int64 out_index = out_index_base + pw;
              /// NOTES(zhengxq): not using the eigen matrix operation for
              /// now.
              for (int d = 0; d < depth; ++d) {
                const T& input_ref = in_mat.coeffRef(d, in_index);
                T& output_ref = out_mat.coeffRef(d, out_index);
                int64& out_arg_max_ref = out_arg_max_mat.coeffRef(d, out_index);
                if (output_ref < input_ref ||
                    out_arg_max_ref == kInvalidMaxPoolingIndex) {
                  output_ref = input_ref;
                  int64 input_offset = in_index * depth + d;
                  out_arg_max_ref = input_offset;
                }
              }
            }
          }
        }
      }
    }

    {
      auto input_backprop_flat = input_backprop->flat<T>();
      auto out_arg_max_flat = output_arg_max->flat<int64>();
      auto out_backprop_flat = out_backprop.flat<T>();

      // Initialize output to 0.
      const int64 in_size = in_rows * in_cols * depth;
      const int64 in_start = start * in_size;
      const int64 in_end = limit * in_size;
      EigenMatrixMap in_shard(input_backprop_flat.data() + in_start, 1,
                              in_end - in_start);
      in_shard.setConstant(T(0));

      // Backpropagate.
      const int out_size = out_height * out_width * depth;
      const int out_start = start * out_size;
      const int out_end = limit * out_size;
      for (int index = out_start; index < out_end; ++index) {
        int input_backprop_index = out_arg_max_flat(index);
        // Although this check is in the inner loop, it is worth its value
        // so we don't end up with memory corruptions. Our benchmark shows that
        // the performance impact is quite small
        CHECK(input_backprop_index >= in_start && input_backprop_index < in_end)
            << "Invalid input backprop index: " << input_backprop_index << ", "
            << in_start << ", " << in_end;
        input_backprop_flat(input_backprop_index) += out_backprop_flat(index);
      }
    }

  };

  const int64 shard_cost = params.tensor_in_rows * params.tensor_in_cols *
                           params.depth * params.window_rows *
                           params.window_cols;
  Shard(worker_threads.num_threads, worker_threads.workers,
        params.tensor_in_batch, shard_cost, shard);
}

REGISTER_KERNEL_BUILDER(
    Name("MaxPool").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    MaxPoolingOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("MaxPool").Device(DEVICE_CPU).TypeConstraint<Eigen::half>("T"),
    MaxPoolingOp<CPUDevice, Eigen::half>);

#if GOOGLE_CUDA
// Forward declarations for the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                            \
  template <>                                                          \
  void SpatialMaxPooling<Eigen::GpuDevice, T>::operator()(             \
      const Eigen::GpuDevice& d, typename TTypes<T, 4>::Tensor output, \
      typename TTypes<T, 4>::ConstTensor input, int window_rows,       \
      int window_cols, int row_stride, int col_stride,                 \
      const Eigen::PaddingType& padding);                              \
  extern template struct SpatialMaxPooling<Eigen::GpuDevice, T>;

DECLARE_GPU_SPEC(float);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Note(jiayq): Currently, the Caffe custom implementation is faster than the
// default Eigen implementation so we are using the custom kernel as the
// default. However, you can explicitly invoke the eigen version using
// kernel_label_map.
REGISTER_KERNEL_BUILDER(Name("MaxPool")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .Label("eigen_tensor"),
                        MaxPoolingOp<Eigen::GpuDevice, float>);
#endif  // GOOGLE_CUDA


#if TENSORFLOW_USE_SYCL

namespace functor {
#define DECLARE_SYCL_SPEC(T)                                           \
  template <>                                                          \
  void SpatialMaxPooling<Eigen::SyclDevice, T>::operator()(            \
      const Eigen::SyclDevice& d, typename TTypes<T, 4>::Tensor output,\
      typename TTypes<T, 4>::ConstTensor input, int window_rows,       \
      int window_cols, int row_stride, int col_stride,                 \
      const Eigen::PaddingType& padding) {}

DECLARE_SYCL_SPEC(float);
#undef DECLARE_SYCL_SPEC
}  // namespace functor

// Note(jiayq): Currently, the Caffe custom implementation is faster than the
// default Eigen implementation so we are using the custom kernel as the
// default. However, you can explicitly invoke the eigen version using
// kernel_label_map.
REGISTER_KERNEL_BUILDER(Name("MaxPool")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<float>("T")
                            .Label("eigen_tensor"),
                        MaxPoolingOp<Eigen::SyclDevice, float>);
#endif  // TENSORFLOW_USE_SYCL


// The operation to compute MaxPool gradients.
// It takes three inputs:
//   - The original input tensor
//   - The original output tensor
//   - Backprop tensor for output
// It produces one output: backprop tensor for input.
template <class Device, class T>
class MaxPoolingGradOp : public OpKernel {
 public:
  explicit MaxPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument("Default MaxPoolinGradOp only supports NHWC ",
                                "on device type ",
                                DeviceTypeString(context->device_type())));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    OP_REQUIRES(
        context, ksize_[3] == 1 && stride_[3] == 1,
        errors::Unimplemented(
            "MaxPoolingGrad is not yet supported on the depth dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(1);
    const Tensor& out_backprop = context->input(2);

    // For maxpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional"));
    OP_REQUIRES(context, tensor_out.dims() == 4,
                errors::InvalidArgument("tensor_out must be 4-dimensional"));
    // For maxpooling, out_backprop should have 4 dimensions.
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    const TensorShape& output_shape = tensor_in.shape();

    Tensor tensor_out_dup;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::v(),
                                          tensor_out.shape(), &tensor_out_dup));
    Tensor tensor_out_arg_max;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int64>::v(),
                                                   tensor_out.shape(),
                                                   &tensor_out_arg_max));

    PoolParameters params{context,  ksize_,      stride_,
                          padding_, FORMAT_NHWC, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    Tensor* output = nullptr;
    if (!context->forward_input_to_output(0, 0, &output)) {
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, output_shape, &output));
    }

    SpatialMaxPoolWithArgMaxHelper<CPUDevice, T>(
        context, &tensor_out_dup, &tensor_out_arg_max, output, tensor_in,
        out_backprop, params, padding_);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

REGISTER_KERNEL_BUILDER(
    Name("MaxPoolGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    MaxPoolingGradOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("MaxPoolGrad").Device(DEVICE_CPU).TypeConstraint<Eigen::half>("T"),
    MaxPoolingGradOp<CPUDevice, Eigen::half>);

#ifdef GOOGLE_CUDA

template <typename T>
static void MaxPoolingBackwardCustomKernel(
    OpKernelContext* context, const std::vector<int32>& size,
    const std::vector<int32>& stride, Padding padding, const Tensor* tensor_in,
    const Tensor& out_backprop, const TensorShape& tensor_in_shape) {
  Tensor* output = nullptr;
  if (!context->forward_input_to_output(0, 0, &output)) {
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, tensor_in_shape, &output));
  }

  PoolParameters params{context, size,        stride,
                        padding, FORMAT_NHWC, tensor_in_shape};
  if (!context->status().ok()) {
    return;
  }

  MaxPoolBackwardNoMask(
      tensor_in->flat<T>().data(), params.tensor_in_batch,
      params.tensor_in_rows, params.tensor_in_cols, params.depth,
      params.out_height, params.out_width, params.window_rows,
      params.window_cols, params.row_stride, params.col_stride, params.pad_rows,
      params.pad_cols, out_backprop.flat<T>().data(),
      output->flat<T>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class MaxPoolingGradOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit MaxPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    const int32 ksize_n = GetTensorDim(ksize_, data_format_, 'N');
    const int32 stride_n = GetTensorDim(stride_, data_format_, 'N');
    OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));

    use_dnn_ = CanUseCudnn();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(1);
    const Tensor& out_backprop = context->input(2);

    // For maxpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional 4"));
    OP_REQUIRES(context, tensor_out.dims() == 4,
                errors::InvalidArgument("tensor_out must be 4-dimensional"));
    // For maxpooling, out_backprop should have 4 dimensions.
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    TensorShape output_shape = tensor_in.shape();

    if (use_dnn_) {
      DnnPoolingGradOp<T>::Compute(
          context, perftools::gputools::dnn::PoolingMode::kMaximum, ksize_,
          stride_, padding_, data_format_, &tensor_in, &tensor_out,
          out_backprop, output_shape);
    } else {
      CHECK(data_format_ == FORMAT_NHWC)
          << "Non-Cudnn MaxPoolGrad only supports NHWC format";
      MaxPoolingBackwardCustomKernel<T>(context, ksize_, stride_, padding_,
                                        &tensor_in, out_backprop, output_shape);
    }
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool use_dnn_;
};

REGISTER_KERNEL_BUILDER(
    Name("MaxPoolGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    MaxPoolingGradOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("MaxPoolGrad").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    MaxPoolingGradOp<Eigen::GpuDevice, Eigen::half>);

#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL

using namespace cl;

template <typename dtype>
class SetZero {
  using write_accessor = sycl::accessor<uint8_t, 1, sycl::access::mode::write, sycl::access::target::global_buffer>;
public:
  SetZero(const int nthreads, write_accessor bottom_diff_access)
      :nthreads_(nthreads)
      ,bottom_diff_access_(bottom_diff_access) {
  }

  void operator()(sycl::nd_item<1> item) {
    dtype* bottom_diff = ConvertToActualTypeSycl(dtype, bottom_diff_access_);
    for (int index = item.get_global(0); index < nthreads_; index += item.get_global_range(0)) {
      *(bottom_diff + index) = dtype(0);
    }
  }

private:
  const int nthreads_;
  write_accessor bottom_diff_access_;
};

template <typename dtype>
class MaxPoolBackwardNoMaskNHWC {
  using atomic_accessor = sycl::accessor<uint8_t, 1, sycl::access::mode::atomic, sycl::access::target::global_buffer>;
  using read_accessor = sycl::accessor<uint8_t, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
 public:
  MaxPoolBackwardNoMaskNHWC(const int nthreads, read_accessor bottom_data, const int height,
                            const int width, const int channels, const int pooled_height,
                            const int pooled_width, const int kernel_h, const int kernel_w,
                            const int stride_h, const int stride_w, const int pad_t, const int pad_l,
                            read_accessor top_diff, atomic_accessor bottom_diff)
      :nthreads_(nthreads)
      ,bottom_data_(bottom_data)
      ,height_(height)
      ,width_(width)
      ,channels_(channels)
      ,pooled_height_(pooled_height)
      ,pooled_width_(pooled_width)
      ,kernel_h_(kernel_h)
      ,kernel_w_(kernel_w)
      ,stride_h_(stride_h)
      ,stride_w_(stride_w)
      ,pad_t_(pad_t)
      ,pad_l_(pad_l)
      ,top_diff_(top_diff)
      ,bottom_diff_(bottom_diff) {
  }

  void operator()(sycl::nd_item<1> item) {
    const dtype* bottom_data = ConvertToActualTypeSycl(dtype, bottom_data_);
    const dtype* top_diff = ConvertToActualTypeSycl(dtype, top_diff_);
    dtype* bottom_diff = ConvertToActualTypeSycl(dtype, bottom_diff_);

    for (int index = item.get_global(0); index < nthreads_; index += item.get_global_range(0)) {
      // First find out the index to the maximum, since we have no mask.
      int n = index;
      int c = n % channels_;
      n /= channels_;
      int wstart = (n % pooled_width_) * stride_w_ - pad_l_;
      n /= pooled_width_;
      int hstart = (n % pooled_height_) * stride_h_ - pad_t_;
      n /= pooled_height_;
      int hend = std::min(hstart + kernel_h_, height_);
      int wend = std::min(wstart + kernel_w_, width_);
      hstart = std::max(hstart, 0);
      wstart = std::max(wstart, 0);
      dtype maxval = Eigen::NumTraits<dtype>::lowest();
      int maxidx = -1;
      const dtype* bottom_data_n = bottom_data + n * height_ * width_ * channels_;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          int idx = (h * width_ + w) * channels_ + c;
          if (bottom_data_n[idx] > maxval) {
            maxidx = idx;
            maxval = bottom_data_n[idx];
          }
        }
      }

      // Atomically accumulate the bottom diff. The index could still be
      // uninitialized, if all the bottom_data are NaN.
      if (maxidx != -1) {
        // TODO make atomic
        *(bottom_diff + n * height_ * width_ * channels_ + maxidx) += top_diff[index];
      }
    }
  }

private:
  const int nthreads_;
  read_accessor bottom_data_;
  const int height_;
  const int width_;
  const int channels_;
  const int pooled_height_;
  const int pooled_width_;
  const int kernel_h_;
  const int kernel_w_;
  const int stride_h_;
  const int stride_w_;
  const int pad_t_;
  const int pad_l_;
  read_accessor top_diff_;
  atomic_accessor bottom_diff_;
};


bool MaxPoolBackwardNoMask(const float* bottom_data, const int batch,
                           const int height, const int width,
                           const int channels, const int pooled_height,
                           const int pooled_width, const int kernel_h,
                           const int kernel_w, const int stride_h,
                           const int stride_w, const int pad_t, const int pad_l,
                           const float* top_diff, float* bottom_diff,
                           const Eigen::SyclDevice& d) {
  const int kThreadsPerBlock = 1024;
  const int bottom_size = batch * channels * height * width;
  const int top_size = batch * channels * pooled_height * pooled_width;

  auto bottom_diff_buffer = d.get_sycl_buffer(bottom_diff);
  auto bottom_data_buffer = d.get_sycl_buffer(bottom_data);
  auto top_diff_buffer = d.get_sycl_buffer(top_diff);

  d.sycl_queue().submit([&](sycl::handler& cgh) {
    auto bottom_diff_access = bottom_diff_buffer.template get_access<sycl::access::mode::write>(cgh);
    SetZero<float> set_zero(bottom_size, bottom_diff_access);

    const size_t group_count = (bottom_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(group_count * kThreadsPerBlock), sycl::range<1>(kThreadsPerBlock)),
      set_zero
    );
  });

  d.sycl_queue().submit([&](sycl::handler& cgh) {
    auto bottom_data_access = bottom_data_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto bottom_diff_access = bottom_diff_buffer.template get_access<sycl::access::mode::atomic>(cgh);
    auto top_diff_access = top_diff_buffer.template get_access<sycl::access::mode::read>(cgh);
    MaxPoolBackwardNoMaskNHWC<float> maxPoolBackward(
      top_size, bottom_data_access, height, width, channels, pooled_height,
      pooled_width, kernel_h, kernel_w, stride_h, stride_w, pad_t, pad_l,
      top_diff_access, bottom_diff_access
    );

    const size_t group_count = (top_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(group_count * kThreadsPerBlock), sycl::range<1>(kThreadsPerBlock)),
      maxPoolBackward
    );
  });

  return d.ok();
}

bool MaxPoolBackwardNoMask(const Eigen::half* bottom_data, const int batch,
                           const int height, const int width,
                           const int channels, const int pooled_height,
                           const int pooled_width, const int kernel_h,
                           const int kernel_w, const int stride_h,
                           const int stride_w, const int pad_t, const int pad_l,
                           const Eigen::half* top_diff, Eigen::half* bottom_diff,
                           const Eigen::SyclDevice& d) {
  const int kThreadsPerBlock = 1024;
  const int bottom_size = batch * channels * height * width;
  const int top_size = batch * channels * pooled_height * pooled_width;

  auto bottom_diff_buffer = d.get_sycl_buffer(bottom_diff);
  auto bottom_data_buffer = d.get_sycl_buffer(bottom_data);
  auto top_diff_buffer = d.get_sycl_buffer(top_diff);

  d.sycl_queue().submit([&](sycl::handler& cgh) {
    auto bottom_diff_access = bottom_diff_buffer.template get_access<sycl::access::mode::write>(cgh);
    const size_t group_count = (bottom_size + kThreadsPerBlock - 1) / kThreadsPerBlock;

    SetZero<Eigen::half> set_zero(bottom_size, bottom_diff_access);
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(group_count * kThreadsPerBlock), sycl::range<1>(kThreadsPerBlock)),
      set_zero
    );
  });

  d.sycl_queue().submit([&](sycl::handler& cgh) {
    auto bottom_data_access = bottom_data_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto bottom_diff_access = bottom_diff_buffer.template get_access<sycl::access::mode::atomic>(cgh);
    auto top_diff_access = top_diff_buffer.template get_access<sycl::access::mode::read>(cgh);
    MaxPoolBackwardNoMaskNHWC<Eigen::half> maxPoolBackward(
      top_size, bottom_data_access, height, width, channels, pooled_height,
      pooled_width, kernel_h, kernel_w, stride_h, stride_w, pad_t, pad_l,
      top_diff_access, bottom_diff_access
    );

    const size_t group_count = (top_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(group_count * kThreadsPerBlock), sycl::range<1>(kThreadsPerBlock)),
      maxPoolBackward
    );
  });

  return d.ok();
}

template <typename T>
static void MaxPoolingBackwardCustomKernel(
    OpKernelContext* context, const std::vector<int32>& size,
    const std::vector<int32>& stride, Padding padding, const Tensor* tensor_in,
    const Tensor& out_backprop, const TensorShape& tensor_in_shape) {
  Tensor* output = nullptr;
  if (!context->forward_input_to_output(0, 0, &output)) {
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, tensor_in_shape, &output));
  }

  PoolParameters params{context, size,        stride,
                        padding, FORMAT_NHWC, tensor_in_shape};
  if (!context->status().ok()) {
    return;
  }

  MaxPoolBackwardNoMask(
      tensor_in->flat<T>().data(), params.tensor_in_batch,
      params.tensor_in_rows, params.tensor_in_cols, params.depth,
      params.out_height, params.out_width, params.window_rows,
      params.window_cols, params.row_stride, params.col_stride, params.pad_rows,
      params.pad_cols, out_backprop.flat<T>().data(),
      output->flat<T>().data(), context->eigen_device<Eigen::SyclDevice>());
}

template <class T>
class MaxPoolingGradOp<Eigen::SyclDevice, T> : public OpKernel {
 public:
  typedef Eigen::SyclDevice Device;

  explicit MaxPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    const int32 ksize_n = GetTensorDim(ksize_, data_format_, 'N');
    const int32 stride_n = GetTensorDim(stride_, data_format_, 'N');
    OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(1);
    const Tensor& out_backprop = context->input(2);

    // For maxpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional 4"));
    OP_REQUIRES(context, tensor_out.dims() == 4,
                errors::InvalidArgument("tensor_out must be 4-dimensional"));
    // For maxpooling, out_backprop should have 4 dimensions.
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    TensorShape output_shape = tensor_in.shape();

    CHECK(data_format_ == FORMAT_NHWC)
        << "SYCL MaxPoolGrad only supports NHWC format";
    MaxPoolingBackwardCustomKernel<T>(context, ksize_, stride_, padding_,
                                    &tensor_in, out_backprop, output_shape);

  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

REGISTER_KERNEL_BUILDER(
    Name("MaxPoolGrad").Device(DEVICE_SYCL).TypeConstraint<float>("T"),
    MaxPoolingGradOp<Eigen::SyclDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("MaxPoolGrad").Device(DEVICE_SYCL).TypeConstraint<Eigen::half>("T"),
    MaxPoolingGradOp<Eigen::SyclDevice, Eigen::half>);

#endif  // TENSORFLOW_USE_SYCL

template <typename Device, typename T>
struct LaunchMaxPoolingNoMask;

template <typename Device, typename T>
class MaxPoolingNoMaskOp : public OpKernel {
 public:
  explicit MaxPoolingNoMaskOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument(
            "Default MaxPoolingNoMaskOp only supports NHWC on device type ",
            DeviceTypeString(context->device_type())));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);

    PoolParameters params{context,  ksize_,       stride_,
                          padding_, data_format_, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape({params.tensor_in_batch, params.out_height,
                           params.out_width, params.depth});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    LaunchMaxPoolingNoMask<Device, T>::launch(context, params, tensor_in,
                                              output);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

template <typename Device, typename T>
struct LaunchMaxPoolingWithArgmax;

template <typename Device, typename T>
class MaxPoolingWithArgmaxOp : public OpKernel {
 public:
  explicit MaxPoolingWithArgmaxOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);

    PoolParameters params{context,  ksize_,      stride_,
                          padding_, FORMAT_NHWC, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape({params.tensor_in_batch, params.out_height,
                           params.out_width, params.depth});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    Tensor* argmax = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, out_shape, &argmax));

    LaunchMaxPoolingWithArgmax<Device, T>::launch(context, params, tensor_in,
                                                  output, argmax);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
};

template <typename Device, typename T>
struct LaunchMaxPoolingGradWithArgmax;

template <typename Device, typename T>
class MaxPoolingGradWithArgmaxOp : public OpKernel {
 public:
  explicit MaxPoolingGradWithArgmaxOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& grad_in = context->input(1);
    const Tensor& argmax = context->input(2);

    PoolParameters params{context,  ksize_,      stride_,
                          padding_, FORMAT_NHWC, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape({params.tensor_in_batch, params.tensor_in_rows,
                           params.tensor_in_cols, params.depth});
    Tensor* grad_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &grad_out));

    LaunchMaxPoolingGradWithArgmax<Device, T>::launch(context, params, grad_in,
                                                      argmax, grad_out);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
};

#if GOOGLE_CUDA
template <typename T>
class MaxPoolingNoMaskOp<GPUDevice, T> : public OpKernel {
 public:
  typedef GPUDevice Device;
  explicit MaxPoolingNoMaskOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    const int32 ksize_n = GetTensorDim(ksize_, data_format_, 'N');
    const int32 stride_n = GetTensorDim(stride_, data_format_, 'N');
    OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    use_dnn_ = CanUseCudnn();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);

    PoolParameters params{context,  ksize_,       stride_,
                          padding_, data_format_, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape =
        ShapeFromFormat(data_format_, params.tensor_in_batch, params.out_height,
                        params.out_width, params.depth);
    if (use_dnn_ && data_format_ == FORMAT_NCHW) {
      DnnPoolingOp<T>::Compute(
          context, perftools::gputools::dnn::PoolingMode::kMaximum, ksize_,
          stride_, padding_, data_format_, tensor_in, out_shape);
    } else {
      CHECK(data_format_ == FORMAT_NHWC)
          << "Non-Cudnn MaxPool only supports NHWC format";
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
      LaunchMaxPoolingNoMask<Device, T>::launch(context, params, tensor_in,
                                                output);
    }
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool use_dnn_;
};

template <typename T>
struct LaunchMaxPoolingNoMask<Eigen::GpuDevice, T> {
  static void launch(OpKernelContext* context, const PoolParameters& params,
                     const Tensor& input, Tensor* output) {
    bool status = MaxPoolForwardWithOptionalArgmax(
        input.flat<T>().data(), params.tensor_in_batch, params.tensor_in_rows,
        params.tensor_in_cols, params.depth, params.out_height,
        params.out_width, params.window_rows, params.window_cols,
        params.row_stride, params.col_stride, params.pad_rows, params.pad_cols,
        output->flat<T>().data(), nullptr, context->eigen_gpu_device());
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching MaxPoolForwardNoMask"));
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("MaxPool").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    MaxPoolingNoMaskOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("MaxPool").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    MaxPoolingNoMaskOp<Eigen::GpuDevice, Eigen::half>);

template <typename T>
struct LaunchMaxPoolingWithArgmax<Eigen::GpuDevice, T> {
  static void launch(OpKernelContext* context, const PoolParameters& params,
                     const Tensor& input, Tensor* output, Tensor* argmax) {
    bool status = MaxPoolForwardWithOptionalArgmax(
        input.flat<T>().data(), params.tensor_in_batch, params.tensor_in_rows,
        params.tensor_in_cols, params.depth, params.out_height,
        params.out_width, params.window_rows, params.window_cols,
        params.row_stride, params.col_stride, params.pad_rows, params.pad_cols,
        output->flat<T>().data(),
        reinterpret_cast<int64*>(argmax->flat<int64>().data()),
        context->eigen_gpu_device());
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching MaxPoolForwardWithArgmax"));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MaxPoolWithArgmax")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int64>("Targmax")
                            .TypeConstraint<float>("T"),
                        MaxPoolingWithArgmaxOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(Name("MaxPoolWithArgmax")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int64>("Targmax")
                            .TypeConstraint<Eigen::half>("T"),
                        MaxPoolingWithArgmaxOp<Eigen::GpuDevice, Eigen::half>);

template <typename T>
struct LaunchMaxPoolingGradWithArgmax<Eigen::GpuDevice, T> {
  static void launch(OpKernelContext* context, const PoolParameters& params,
                     const Tensor& grad_in, const Tensor& argmax,
                     Tensor* grad_out) {
    const int input_size = params.tensor_in_batch * params.tensor_in_rows *
                           params.tensor_in_cols * params.depth;
    const int output_size = params.tensor_in_batch * params.out_height *
                            params.out_width * params.depth;
    const int top_offset = params.out_height * params.out_width * params.depth;
    const int bottom_offset =
        params.tensor_in_rows * params.tensor_in_cols * params.depth;
    bool status = MaxPoolBackwardWithArgmax(
        output_size, input_size, grad_in.flat<T>().data(),
        reinterpret_cast<const int64*>(argmax.flat<int64>().data()), top_offset,
        bottom_offset, grad_out->flat<T>().data(), context->eigen_gpu_device());
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching MaxPoolForwardWithArgmax"));
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("MaxPoolGradWithArgmax")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T")
        .TypeConstraint<int64>("Targmax"),
    MaxPoolingGradWithArgmaxOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("MaxPoolGradWithArgmax")
        .Device(DEVICE_GPU)
        .TypeConstraint<Eigen::half>("T")
        .TypeConstraint<int64>("Targmax"),
    MaxPoolingGradWithArgmaxOp<Eigen::GpuDevice, Eigen::half>);

#endif  // GOOGLE_CUDA

#if TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;

template <typename T>
class MaxPoolingNoMaskOp<SYCLDevice, T> : public OpKernel {
 public:
  typedef SYCLDevice Device;
  explicit MaxPoolingNoMaskOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    const int32 ksize_n = GetTensorDim(ksize_, data_format_, 'N');
    const int32 stride_n = GetTensorDim(stride_, data_format_, 'N');
    OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);

    PoolParameters params{context,  ksize_,       stride_,
                          padding_, data_format_, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape =
        ShapeFromFormat(data_format_, params.tensor_in_batch, params.out_height,
                        params.out_width, params.depth);

    CHECK(data_format_ == FORMAT_NHWC)
        << "SYCL MaxPool only supports NHWC format";
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    LaunchMaxPoolingNoMask<Device, T>::launch(context, params, tensor_in,
                                              output);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

template <typename dtype, typename mask_dtype>
class MaxPoolForwardNHWC {
  using write_accessor = sycl::accessor<uint8_t, 1, sycl::access::mode::write, sycl::access::target::global_buffer>;
  using read_accessor = sycl::accessor<uint8_t, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
 public:
  MaxPoolForwardNHWC(const int nthreads, read_accessor bottom_data,
                     const int height, const int width,
                     const int channels, const int pooled_height,
                     const int pooled_width, const int kernel_h,
                     const int kernel_w, const int stride_h,
                     const int stride_w, const int pad_t,
                     const int pad_l, write_accessor top_data,
                     write_accessor mask)
      :nthreads_(nthreads)
      ,bottom_data_(bottom_data)
      ,height_(height)
      ,width_(width)
      ,channels_(channels)
      ,pooled_height_(pooled_height)
      ,pooled_width_(pooled_width)
      ,kernel_h_(kernel_h)
      ,kernel_w_(kernel_w)
      ,stride_h_(stride_h)
      ,stride_w_(stride_w)
      ,pad_t_(pad_t)
      ,pad_l_(pad_l)
      ,top_data_(top_data)
      ,mask_(mask) {
  }

  void operator()(sycl::nd_item<1> item) {
    dtype* bottom_data = ConvertToActualTypeSycl(dtype, bottom_data_);
    dtype* top_data = ConvertToActualTypeSycl(dtype, top_data_);
    mask_dtype* mask = ConvertToActualTypeSycl(mask_dtype, mask_);

    for (int index = item.get_global(0); index < nthreads_; index += item.get_global_range(0)) {
      int n = index;
      int c = n % channels_;
      n /= channels_;
      int wstart = (n % pooled_width_) * stride_w_ - pad_l_;
      n /= pooled_width_;
      int hstart = (n % pooled_height_) * stride_h_ - pad_t_;
      n /= pooled_height_;
      int hend = std::min(hstart + kernel_h_, height_);
      int wend = std::min(wstart + kernel_w_, width_);
      hstart = std::max(hstart, 0);
      wstart = std::max(wstart, 0);
      dtype maxval = Eigen::NumTraits<dtype>::lowest();
      int maxidx = -1;
      const dtype* bottom_data_n = bottom_data + n * height_ * width_ * channels_;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          int idx = (h * width_ + w) * channels_ + c;
          if (bottom_data_n[idx] > maxval) {
            maxidx = idx;
            maxval = bottom_data_n[idx];
          }
        }
      }
      top_data[index] = maxval;
      if (mask != nullptr) {
        mask[index] = maxidx;
      }
    }
  }

 private:
  const int nthreads_;
  read_accessor bottom_data_;
  const int height_;
  const int width_;
  const int channels_;
  const int pooled_height_;
  const int pooled_width_;
  const int kernel_h_;
  const int kernel_w_;
  const int stride_h_;
  const int stride_w_;
  const int pad_t_;
  const int pad_l_;
  write_accessor top_data_;
  write_accessor mask_;
};

// Run the forward pass of max pooling, optionally writing the argmax indices to
// the mask array, if it is not nullptr. If mask is passed in as nullptr, the
// argmax indices are not written.
bool MaxPoolForwardWithOptionalArgmax(
    const float* bottom_data, const int batch, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_t, const int pad_l,
    float* top_data, int64* mask, const Eigen::SyclDevice& device)
{
  const int kThreadsPerBlock = 1024;
  const int output_size = batch * channels * pooled_height * pooled_width;

  auto bottom_buffer = device.get_sycl_buffer(bottom_data);
  auto top_buffer = device.get_sycl_buffer(top_data);
  auto mask_buffer = device.get_sycl_buffer(mask);

  device.sycl_queue().submit([&](sycl::handler& cgh) {
    auto bottom_access = bottom_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto top_access = bottom_buffer.template get_access<sycl::access::mode::write>(cgh);
    auto mask_access = mask_buffer.template get_access<sycl::access::mode::write>(cgh);
    MaxPoolForwardNHWC<float, int64> maxPool(
      output_size, bottom_access, height, width, channels, pooled_height,
      pooled_width, kernel_h, kernel_w, stride_h, stride_w, pad_t, pad_l,
      top_access, mask_access
    );

    const size_t group_count = (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    cgh.parallel_for<class MaxPoolForwardNHWC<float, int64>>(
      sycl::nd_range<1>(sycl::range<1>(group_count * kThreadsPerBlock), sycl::range<1>(kThreadsPerBlock)),
      maxPool
    );
  });

  return device.ok();
}

bool MaxPoolForwardWithOptionalArgmax(
    const Eigen::half* bottom_data, const int batch, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_t, const int pad_l,
    Eigen::half* top_data, int64* mask, const Eigen::SyclDevice& device) {

  const int kThreadsPerBlock = 1024;
  const int output_size = batch * channels * pooled_height * pooled_width;

  auto bottom_buffer = device.get_sycl_buffer(bottom_data);
  auto top_buffer = device.get_sycl_buffer(top_data);
  auto mask_buffer = device.get_sycl_buffer(mask);

  device.sycl_queue().submit([&](sycl::handler& cgh) {
    auto bottom_access = bottom_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto top_access = bottom_buffer.template get_access<sycl::access::mode::write>(cgh);
    auto mask_access = mask_buffer.template get_access<sycl::access::mode::write>(cgh);

    MaxPoolForwardNHWC<Eigen::half, int64> maxPool(
      output_size, bottom_access, height, width, channels, pooled_height,
      pooled_width, kernel_h, kernel_w, stride_h, stride_w, pad_t, pad_l,
      top_access, mask_access
    );

    const size_t group_count = (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    cgh.parallel_for<class MaxPoolForwardNHWC<Eigen::half, int64>>(
      sycl::nd_range<1>(sycl::range<1>(group_count * kThreadsPerBlock), sycl::range<1>(kThreadsPerBlock)),
      maxPool
    );
  });

  return device.ok();
}

template <typename dtype, typename mask_dtype>
class MaxPoolBackward {
  using atomic_accessor = sycl::accessor<uint8_t, 1, sycl::access::mode::atomic, sycl::access::target::global_buffer>;
  using read_accessor = sycl::accessor<uint8_t, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
public:
  MaxPoolBackward(const int nthreads, read_accessor top_diff,
                  read_accessor mask, const int top_offset,
                  const int bottom_offset, atomic_accessor bottom_diff)
      :nthreads_(nthreads)
      ,top_diff_(top_diff)
      ,mask_(mask)
      ,top_offset_(top_offset)
      ,bottom_offset_(bottom_offset)
      ,bottom_diff_(bottom_diff) {
  }

  void operator()(sycl::nd_item<1> item) {
    const dtype* top_diff = ConvertToActualTypeSycl(dtype, top_diff_);
    const mask_dtype* mask = ConvertToActualTypeSycl(mask_dtype, mask_);
    dtype* bottom_diff = ConvertToActualTypeSycl(dtype, bottom_diff_);
    for (int index = item.get_global(0); index < nthreads_; index += item.get_num_groups(0) * item.get_local_range()[0]) {
      int image_id = (index / top_offset_);
      // TODO make atomic
      *(bottom_diff + image_id * bottom_offset_ + mask[index]) += top_diff[index];
    }
  }

 private:
  const int nthreads_;
  read_accessor top_diff_;
  read_accessor mask_;
  const int top_offset_;
  const int bottom_offset_;
  atomic_accessor bottom_diff_;
};

bool MaxPoolBackwardWithArgmax(const int output_size, const int input_size,
                               const float* top_diff, const int64* mask,
                               const int top_offset, const int bottom_offset,
                               float* bottom_diff, const Eigen::SyclDevice& d) {

  const int kThreadsPerBlock = 1024;

  auto bottom_buffer = d.get_sycl_buffer(bottom_diff);
  d.sycl_queue().submit([&](sycl::handler& cgh) {
    auto bottom_access = bottom_buffer.template get_access<sycl::access::mode::write>(cgh);
    SetZero<float> setZero(input_size, bottom_access);

    const size_t group_count = (input_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    cgh.parallel_for<class SetZero<float>>(
      sycl::nd_range<1>(sycl::range<1>(group_count * kThreadsPerBlock), sycl::range<1>(kThreadsPerBlock)),
      setZero
    );
  });

  auto mask_buffer = d.get_sycl_buffer(mask);
  auto top_diff_buffer = d.get_sycl_buffer(top_diff);
  d.sycl_queue().submit([&](sycl::handler& cgh) {
    auto bottom_access = bottom_buffer.template get_access<sycl::access::mode::atomic>(cgh);
    auto mask_access = mask_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto top_access = top_diff_buffer.template get_access<sycl::access::mode::read>(cgh);
    MaxPoolBackward<float, int64> maxPoolBackward(output_size, top_access, mask_access, top_offset, bottom_offset, bottom_access);

    const size_t group_count = (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    cgh.parallel_for<class MaxPoolBackward<float, int64>>(
      sycl::nd_range<1>(sycl::range<1>(group_count * kThreadsPerBlock), sycl::range<1>(kThreadsPerBlock)),
      maxPoolBackward
    );
  });

  return d.ok();
}

bool MaxPoolBackwardWithArgmax(const int output_size, const int input_size,
                               const Eigen::half* top_diff, const int64* mask,
                               const int top_offset, const int bottom_offset,
                               Eigen::half* bottom_diff,
                               const Eigen::SyclDevice& d) {

  const int kThreadsPerBlock = 1024;

  auto bottom_buffer = d.get_sycl_buffer(bottom_diff);
  d.sycl_queue().submit([&](sycl::handler& cgh) {
    auto bottom_access = bottom_buffer.template get_access<sycl::access::mode::write>(cgh);
    SetZero<Eigen::half> setZero(input_size, bottom_access);

    const size_t group_count = (input_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    cgh.parallel_for<class SetZero<Eigen::half>>(
      sycl::nd_range<1>(sycl::range<1>(group_count * kThreadsPerBlock), sycl::range<1>(kThreadsPerBlock)),
      setZero
    );
  });

  auto mask_buffer = d.get_sycl_buffer(mask);
  auto top_diff_buffer = d.get_sycl_buffer(top_diff);
  d.sycl_queue().submit([&](sycl::handler& cgh) {
    auto bottom_access = bottom_buffer.template get_access<sycl::access::mode::atomic>(cgh);
    auto mask_access = mask_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto top_access = top_diff_buffer.template get_access<sycl::access::mode::read>(cgh);
    MaxPoolBackward<Eigen::half, int64> maxPollBackward(output_size, top_access, mask_access, top_offset, bottom_offset, bottom_access);


    const size_t group_count = (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    cgh.parallel_for<class MaxPoolBackward<Eigen::half, int64>>(
      sycl::nd_range<1>(sycl::range<1>(group_count * kThreadsPerBlock), sycl::range<1>(kThreadsPerBlock)),
      maxPollBackward
    );
  });

  return d.ok();
}

template <typename T>
struct LaunchMaxPoolingNoMask<Eigen::SyclDevice, T> {
  static void launch(OpKernelContext* context, const PoolParameters& params,
                     const Tensor& input, Tensor* output) {
    bool status = MaxPoolForwardWithOptionalArgmax(
        input.flat<T>().data(), params.tensor_in_batch, params.tensor_in_rows,
        params.tensor_in_cols, params.depth, params.out_height,
        params.out_width, params.window_rows, params.window_cols,
        params.row_stride, params.col_stride, params.pad_rows, params.pad_cols,
        output->flat<T>().data(), nullptr, context->eigen_sycl_device());
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching MaxPoolForwardNoMask"));
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("MaxPool").Device(DEVICE_SYCL).TypeConstraint<float>("T"),
    MaxPoolingNoMaskOp<Eigen::SyclDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("MaxPool").Device(DEVICE_SYCL).TypeConstraint<Eigen::half>("T"),
    MaxPoolingNoMaskOp<Eigen::SyclDevice, Eigen::half>);

template <typename T>
struct LaunchMaxPoolingWithArgmax<Eigen::SyclDevice, T> {
  static void launch(OpKernelContext* context, const PoolParameters& params,
                     const Tensor& input, Tensor* output, Tensor* argmax) {
    bool status = MaxPoolForwardWithOptionalArgmax(
        input.flat<T>().data(), params.tensor_in_batch, params.tensor_in_rows,
        params.tensor_in_cols, params.depth, params.out_height,
        params.out_width, params.window_rows, params.window_cols,
        params.row_stride, params.col_stride, params.pad_rows, params.pad_cols,
        output->flat<T>().data(),
        reinterpret_cast<int64*>(argmax->flat<int64>().data()),
        context->eigen_sycl_device());
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching MaxPoolForwardWithArgmax"));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MaxPoolWithArgmax")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int64>("Targmax")
                            .TypeConstraint<float>("T"),
                        MaxPoolingWithArgmaxOp<Eigen::SyclDevice, float>);
REGISTER_KERNEL_BUILDER(Name("MaxPoolWithArgmax")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int64>("Targmax")
                            .TypeConstraint<Eigen::half>("T"),
                        MaxPoolingWithArgmaxOp<Eigen::SyclDevice, Eigen::half>);

template <typename T>
struct LaunchMaxPoolingGradWithArgmax<Eigen::SyclDevice, T> {
  static void launch(OpKernelContext* context, const PoolParameters& params,
                     const Tensor& grad_in, const Tensor& argmax,
                     Tensor* grad_out) {
    const int input_size = params.tensor_in_batch * params.tensor_in_rows *
                           params.tensor_in_cols * params.depth;
    const int output_size = params.tensor_in_batch * params.out_height *
                            params.out_width * params.depth;
    const int top_offset = params.out_height * params.out_width * params.depth;
    const int bottom_offset =
        params.tensor_in_rows * params.tensor_in_cols * params.depth;
    bool status = MaxPoolBackwardWithArgmax(
        output_size, input_size, grad_in.flat<T>().data(),
        reinterpret_cast<const int64*>(argmax.flat<int64>().data()), top_offset,
        bottom_offset, grad_out->flat<T>().data(), context->eigen_sycl_device());
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching MaxPoolForwardWithArgmax"));
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("MaxPoolGradWithArgmax")
        .Device(DEVICE_SYCL)
        .TypeConstraint<float>("T")
        .TypeConstraint<int64>("Targmax"),
    MaxPoolingGradWithArgmaxOp<Eigen::SyclDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("MaxPoolGradWithArgmax")
        .Device(DEVICE_SYCL)
        .TypeConstraint<Eigen::half>("T")
        .TypeConstraint<int64>("Targmax"),
    MaxPoolingGradWithArgmaxOp<Eigen::SyclDevice, Eigen::half>);

#endif  // TENSORFLOW_USE_SYCL

}  // namespace tensorflow
