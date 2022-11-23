/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRANSPOSE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRANSPOSE_CPU_KERNEL_H_

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "nnacl/base/transpose_base.h"

namespace mindspore {
namespace kernel {
/**
 * Transpose算子的CPU实现，继承于CPU算子的父类`DeprecatedNativeCpuKernelMod`
 */
class TransposeFwdCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  // 使用默认的构造函数和析构函数
  TransposeFwdCpuKernelMod() = default;
  ~TransposeFwdCpuKernelMod() override = default;
  // 算子初始化函数
  void InitKernel(const CNodePtr &kernel_node) override;
  // CPU算子入口
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  // Transpose算子的实际入口
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  // 使用并行计算的方法对Transpose算子进行加速
  template <typename T>
  void ParallelRun(const T *input_addr, T *output_addr, const int *output_shape, size_t count);
  // 储存输入、输出的维度等信息
  TransposeParameter transpose_param_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> axes_;
  TypeId dtype_{kTypeUnknown};
  using TypeKernel =
    std::function<void(TransposeFwdCpuKernelMod *, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &)>;
  std::unordered_map<TypeId, TypeKernel> launch_map_;
  TypeKernel launch_func_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRANSPOSE_CPU_KERNEL_H_
