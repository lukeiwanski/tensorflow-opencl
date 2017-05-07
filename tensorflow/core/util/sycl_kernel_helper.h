/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_SYCL_KERNEL_HELPER_H_
#define TENSORFLOW_CORE_UTIL_SYCL_KERNEL_HELPER_H_

#include <algorithm>

#ifdef TENSORFLOW_USE_SYCL

namespace tensorflow {

template <typename dtype, typename write_accessor>
class SetZero {
public:
  SetZero(const int nthreads, write_accessor bottom_diff_access)
      :nthreads_(nthreads)
      ,bottom_diff_access_(bottom_diff_access) {
  }

  void operator()(cl::sycl::nd_item<1> item) {
    dtype* bottom_diff = ConvertToActualTypeSycl(dtype, bottom_diff_access_);
    for (int index = item.get_global(0); index < nthreads_; index += item.get_global_range(0)) {
      *(bottom_diff + index) = dtype(0);
    }
  }

private:
  const int nthreads_;
  write_accessor bottom_diff_access_;
};

template <typename BinaryOperation>
void SyclAtomicOperation(cl::sycl::atomic<uint32_t> element, float value) {
  union {
    uint32_t u32;
    float f32;
  } next, expected;

  BinaryOperation operation;
  expected.u32 = element.load();
  do {
    next.f32 = operation(expected.f32, value);
  } while (element.compare_exchange_strong(expected.u32, next.u32,
           std::memory_order_relaxed,
           std::memory_order_relaxed));
}

}

#endif // TENSORFLOW_USE_SYCL

#endif  // TENSORFLOW_CORE_UTIL_SYCL_KERNEL_HELPER_H_