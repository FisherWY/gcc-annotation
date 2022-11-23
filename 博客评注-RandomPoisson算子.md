# 算子介绍

## 定义

泊松分布是一种统计与概率学里常见的离散概率分布，适合描述单位时间内随机事件发生的次数的概率分布，如：某一服务设施在一定时间内受到的服务请求的次数、电话交换机接到呼叫的次数、汽车站台的候车人数等。

泊松分布的函数表达式为：

$$P(X=k) = \frac{e^{-\Lambda}\Lambda^k}{k!}$$

其中，参数$\Lambda$是随机事件发生次数的数学期望值。若$X$服从参数为$\Lambda$的泊松分布，记为$X \sim Pois(\Lambda)$。

关于泊松分布的更多定义，可以参考Wikipedia，[链接](https://zh.wikipedia.org/wiki/%E5%8D%9C%E7%93%A6%E6%9D%BE%E5%88%86%E5%B8%83)。

## 算子功能

给定描述待采样矩阵维度的1-D Tensor: `shape`，以及期望值`rate`（对应泊松分布定义中的$\Lambda$），算子根据输入的期望值`rate`以及待采样矩阵`shape`，生成随机泊松分布的采样结果，采样结果的维度由给定的`shape`与期望值`rate`的维度拼接而成。比如说，输入的待采样矩阵的维度`shape`为`[3, 3]`，输入的期望值`rate`维度为5，则输出的结果维度为`[3, 3, 5]`。

# 源码分析

## Python侧算子原语定义

在`Mindspore`中，所有算子都使用算子原语（Primitive）进行封装，为底层`Ascend`，`GPU`，`AICPU`，`CPU`等设备的算子具体实现提供统一的调用接口。在Python侧的算子定义中，通常只需要实现类最基本的初始化即可，也就是实现`__init__`函数，在初始化过程中需要对初始化的参数进行合法性检查，在文件`mindspore/python/mindspore/ops/operations/random_ops.py`中，`RandomPoisson`算子的Python侧原语定义如下。

```python
class RandomPoisson(Primitive):
    r"""
    Produces random non-negative  values i, distributed according to discrete probability function:

    .. math::
        \text{P}(i|μ) = \frac{\exp(-μ)μ^{i}}{i!},

    Args:
         seed (int): An optional int. Defaults to 0. If either `seed` or `seed2` are set to be non-zero,
            the seed is set by the given seed. Otherwise, it is seeded by a random seed.
         seed2 (int): An optional int. Defaults to 0. A second seed to avoid seed collision.
         dtype (mindspore.dtype): The type of output. Default: mindspore.int64.

    Inputs:
        - **shape** (Tensor) - The shape of random tensor to be generated, 1-D Tensor, whose dtype must be in
                               [int32, int64]
        - **rate** (Tensor) - μ parameter the distribution was constructed with. The parameter defines mean number
          of occurrences of the event. Its type must be in [float16, float32, float64, int32, int64]

    Outputs:
        Tensor. Its shape is (*shape, *rate.shape). Its type is spcified by `dtype`.

    Raises:
        TypeError: If `shape` is not a Tensor or its dtype is not int32 or int64.
        TypeError: If `dtype` is not int32 or int64.
        ValueError: If `shape` is not a 1-D tensor.
        ValueError: If `shape` elements are negative.

    Supported Platforms:
        ``Ascend````GPU````CPU``

    Examples:
        >>> shape = Tensor(np.array([2, 3]), mstype.int32)
        >>> rate = Tensor(np.array([2, 2]), mstype.int32)
        >>> seed = 0
        >>> seed2 = 0
        >>> random_poisson = ops.RandomPoisson(seed=seed, seed2=seed2)
        >>> output = random_poisson(shape,rate)
        >>> print(output.shape)
        (2, 3, 2)
    """

    @prim_attr_register
    def __init__(self, seed=0, seed2=0, dtype=mstype.int64):
        """Initialize Poisson"""
        self.init_prim_io_names(inputs=['shape', 'rate'], outputs=['output'])
        Validator.check_value_type('seed', seed, [int], self.name)
        Validator.check_value_type('seed2', seed2, [int], self.name)
        valid_values = (mstype.int64, mstype.int32, mstype.float16, mstype.float32, mstype.float64)
        Validator.check_type_name("dtype", dtype, valid_values, self.name)
```

每一段代码对应的作用如下：
- 首行：定义泊松算子类，继承于算子原语`Primitive`类。
- 注释部分：算子文档，描述算子功能、算子参数、算子输入、算子输出、抛出错误的类型、支持的平台、代码样例，该注释也用于生成`Mindspore`在线文档。
- __init__函数：算子初始化，`init_prim_io_names`方法向`Mindspore`框架注册该算子的输入名称和输出名称，`Validator`类用于检查输入参数是否合法。

## C++侧算子注册

在C++侧，算子首先需要将自己注册到`Mindspore`框架的全局变量中，来“告知”框架，存在一个名字叫`RandomPoisson`的算子，算子在`mindspore/core/ops/core_ops.h`中注册，代码如下。

```c++
GVAR_DEF(PrimitivePtr, kPrimRandomPoisson, std::make_shared<Primitive>("RandomPoisson"));
```

## C++侧算子原语定义

算子在C++侧也需要定义自己的`Primitive`原语，用于算子在C++侧的初始化、参数合法性检查等。C++侧的原语定义分为`mindspore/core/ops/random_poisson.h`和`mindspore/core/ops/random_poisson.cc`两个文件，头文件主要是定义算子类，cc文件主要是类成员函数的具体实现，由于源码代码比较长，因此将代码实现具体功能的描述都写在注释中。

```c++
// 定义随机泊松算子原语类
#ifndef MINDSPORE_CORE_OPS_RANDOM_POISSON_H_
#define MINDSPORE_CORE_OPS_RANDOM_POISSON_H_
// C++标准库
#include <map>
#include <vector>
#include <string>
#include <memory>
// Mindspore算子原语父类
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

// Mindspore命名空间域，算子需要定义在`mindspore::ops`中
namespace mindspore {
namespace ops {
// 随机泊松算子的名称
constexpr auto kRandomPoisson = "RandomPoisson";
// 算子类定义，继承于`BaseOperator`
class MIND_API RandomPoisson : public BaseOperator {
 public:
  // 使用宏定义，定义使用智能指针作为参数的构造函数和默认的析构函数
  MIND_API_BASE_MEMBER(RandomPoisson);
  // 类构造函数，构造算子的输入、输出
  RandomPoisson() : BaseOperator(kRandomPoisson) { InitIOName({"shape", "rate"}, {"output"}); }
  // 实现父类`Init`方法，并没有实际作用
  void Init() const {}
  // 算子随机种子`seed`的setter和getter方法
  void set_seed(const int64_t seed);
  int64_t get_seed() const;
  void set_seed2(const int64_t seed2);
  int64_t get_seed2() const;
};
// 算子执行计算的正式入口
abstract::AbstractBasePtr RandomPoissonInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args);
// 指向算子的智能指针
using kPrimPrimRandomPoissonPtr = std::shared_ptr<RandomPoisson>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RANDOM_POISSON_H_
```

```c++
// 随机泊松算子的头文件
#include "ops/random_poisson.h"
// C++标准库
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include <map>
// Mindspore公共工具类函数库
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "mindapi/src/helper.h"
// 与头文件相同的Mindspore命名空间域
namespace mindspore {
namespace ops {
// 自定义的命名空间域，将随机泊松算子需要的一些私有函数定义在该空间域中，以防止出现函数重名冲突
namespace {
// 对算子的输入进行维度合法性检查
abstract::ShapePtr RandomPoissonInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  // Mindspore公有工具函数，检查传入的参数是否为空指针
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  // 检查输入参数`shape`的维度，该参数只能为1-D Tensor
  auto shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  if (shape_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For RandomPoisson, the argument[shape] must be a 1-D tensor, but got "
                             << shape_shape.size() << "-D";
  }
  // 检查输入参数`shape`的值，如果值为空则抛出错误
  auto shape_value = input_args[kInputIndex0]->BuildValue();
  MS_EXCEPTION_IF_NULL(shape_value);
  if (!shape_value->isa<AnyValue>() && !shape_value->isa<None>()) {
    auto out_shape = CheckAndConvertUtils::CheckTensorIntValue("shape", shape_value, op_name);
    (void)CheckAndConvertUtils::CheckPositiveVector("shape", out_shape, op_name);
    // 将`shape`的值和`rate`的维度拼接起来，构造算子最终输出结果的维度
    auto rate_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
    auto rate_rank = SizeToLong(rate_shape.size());
    for (int64_t i = 0; i < rate_rank; i++) {
      out_shape.push_back(rate_shape[i]);
    }

    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    // 当输入参数`shape`的值为空时，则没有输出，在算子具体执行时会抛出错误
    std::vector<int64_t> output_shape = {-2};
    ShapeVector shape_min = {1};
    ShapeVector shape_max = {1};
    return std::make_shared<abstract::Shape>(output_shape, shape_min, shape_max);
  }
}

// 对算子输入参数的数据类型进行合法性检查
TypePtr RandomPoissonInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  const std::set<TypePtr> valid_shape_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTypeValid("shape", input_args[0]->BuildType(), valid_shape_types, prim_name);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTypeValid("rate", input_args[1]->BuildType(), valid_types, prim_name);
  auto dtype_value = prim->GetAttr("dtype");
  if (!dtype_value->isa<Type>()) {
    MS_EXCEPTION(TypeError) << "For RandomPoisson, the dtype of " + prim_name + " is invalid!";
  }
  auto output_type = dtype_value->cast<TypePtr>();
  return CheckAndConvertUtils::CheckSubClass("dtype", output_type, valid_types, prim_name);
}
}  // namespace

// 第一个随机种子seed的getter方法
int64_t RandomPoisson::get_seed() const {
  auto value_ptr = this->GetAttr(kSeed);
  return GetValue<int64_t>(value_ptr);
}

// 第一个随机种子seed的setter方法
void RandomPoisson::set_seed(const int64_t seed) { (void)this->AddAttr(kSeed, api::MakeValue(seed)); }

// 第二个随机种子seed2的getter方法
int64_t RandomPoisson::get_seed2() const {
  auto value_ptr = this->GetAttr(kSeed2);
  return GetValue<int64_t>(value_ptr);
}

// 第二个随机种子seed2的setter方法
void RandomPoisson::set_seed2(const int64_t seed2) { (void)this->AddAttr(kSeed2, api::MakeValue(seed2)); }

// 算子执行计算的正式入口，调用以上的`RandomPoissonInferType`和`RandomPoissonInferShape`进行参数合法性检查后，正式开始算子的执行
AbstractBasePtr RandomPoissonInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infertype = RandomPoissonInferType(primitive, input_args);
  auto infershape = RandomPoissonInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}
// 注册一些需要在Host端直接获取的数据
REGISTER_HOST_DEPENDS(kRandomPoisson, {0});
// 向Mindspore框架注册算子
MIND_API_OPERATOR_IMPL(RandomPoisson, BaseOperator);
// 向Mindspore框架注册算子入口
REGISTER_PRIMITIVE_EVAL_IMPL(RandomPoisson, prim::kPrimRandomPoisson, RandomPoissonInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
```

## C++侧GPU算子实现

光是定义了算子的原语还是无法实现完整的功能的，接下来还需要实现算子在不同类型设备上的执行逻辑，由于我的任务是实现GPU算子，因此在这里只分析GPU算子的源码。GPU算子又分为C++实现和CUDA实现两个部分，C++侧负责算子执行前的一些准备工作，CUDA侧主要利用GPU并行计算的能力加速算子的计算。

首先介绍算子的C++侧实现，算子核函数类同样分为头文件`mindspore/ccsrc/plugin/device/gpu/kernel/random/random_poisson_gpu_kernel.h`和具体实现`mindspore/ccsrc/plugin/device/gpu/kernel/random/random_poisson_gpu_kernel.cc`。由于代码比较长，因此以注释的方式去分析和解读代码。

```c++
// C++侧算子核函数的头文件定义
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDOM_POISSON_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDOM_POISSON_GPU_KERNEL_H_
// CUDA相关头文件
#include <curand_kernel.h>
#include <cuda_runtime_api.h>
// C++标准库
#include <vector>
#include <map>
#include <string>
#include <utility>
// Mindspore的GPU算子Kernel有关的头文件
#include "mindspore/core/ops/random_poisson.h"
#include "kernel/common_utils.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/random_op_impl.cuh"
// Mindspore将GPU Kernel的定义放在命名空间域`mindspore::kernel`中
namespace mindspore {
namespace kernel {
// GPU算子类的定义，继承于`NativeGpuKernelMod`类，使用Helper类`MatchKernelHelper`辅助进行参数类型合法性检查、算子支持数据类型注册
class RandomPoissonGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<RandomPoissonGpuKernelMod> {
 public:
  // GPU Kernel的默认构造函数和析构函数
  RandomPoissonGpuKernelMod() = default;
  ~RandomPoissonGpuKernelMod() override = default;
  // 算子初始化，该函数在类的构造过程中只执行一次，可以在函数中设置一些非输入输出类的参数的值，如：随机种子seed等
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  // 算子的运行入口，Mindspore为算子分配GPU的device_id，给出输入、工作空间、输出的device端地址
  // 这里调用Helper类的`kernel_func`函数，Helper类对所有参数进行非空检查，检查通过后调用`LaunchKernel`函数正式执行算子
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }
  // 该方法主要用于计算算子在本次运行中所需要的内存空间大小，该方法仅仅是计算需要的内存空间大小，而不进行实际的内存分配
  // 计算结果的单位为`byte`，因此通常计算结果大小是`element_count * sizeof(element)`
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;
  // Helper类函数的实现，将算子支持的所有数据类型组合并构造成支持列表
  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  // 重设算子资源，通常在`Resize`函数中使用
  void ResetResource() noexcept {
    rate_elements_ = 1;
    output_elements_ = 1;
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }
  // 用于向Mindspore框架返回算子支持的数据类型，与Helper类的`OpSupport`和`GetFuncList`函数配合使用
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  // 算子核函数执行方法，在该方法中调用CUDA核函数，使用GPU加速算子计算
  template <typename R, typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);
  // 算子私有变量，主要储存随机种子seed的值，和各个输入元素所占内存空间大小
  int64_t rate_elements_;
  int64_t output_elements_;
  int64_t unit_shape_size_;
  int64_t unit_rate_size_;
  int64_t unit_output_size_;
  int64_t seed_{0};
  int64_t seed2_{0};
  bool is_null_input_{false};
  void *cuda_stream_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDOM_POISSON_GPU_KERNEL_H_
```

```c++
// 算子GPU Kernel头文件
#include "plugin/device/gpu/kernel/random/random_poisson_gpu_kernel.h"
// C++标准库
#include <functional>
#include <utility>
#include <memory>
#include <string>
#include <algorithm>
// Mindspore工具类
#include "ir/anf.h"
#include "utils/log_adapter.h"
#include "kernel/common_utils.h"
// CUDA FP16头文件
#include "include/cuda_fp16.h"
// 命名空间域`mindspore::kernel`
namespace mindspore {
namespace kernel {
// 匿名命名空间域，存放只有本算子使用的宏定义，用于注册算子支持的数据类型
namespace {
using KernelRunFunc = RandomPoissonGpuKernelMod::KernelRunFunc;
#define ADD_KERNEL(shape_dtype, rate_dtype, output_dtype, rate_type, output_type) \
  {                                                                               \
    KernelAttr()                                                                  \
      .AddInputAttr(kNumberType##shape_dtype)                                     \
      .AddInputAttr(kNumberType##rate_dtype)                                      \
      .AddOutputAttr(kNumberType##output_dtype),                                  \
      &RandomPoissonGpuKernelMod::LaunchKernel<rate_type, output_type>            \
  }
}  // namespace
// 算子初始化函数，只会在类构造时调用一次
bool RandomPoissonGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  // 使用Helper函数对参数进行合法性检查
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  // 获取各种参数类型所占的内存空间，用于在`Resize`函数中计算算子运行过程中所需内存大小
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  unit_shape_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(0).first);
  unit_rate_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(1).first);
  unit_output_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(0).first);
  // 获取算子的随机种子seed、seed2
  auto kernel_ptr = std::make_shared<ops::RandomPoisson>(base_operator->GetPrim());
  seed_ = static_cast<int64_t>(kernel_ptr->get_seed());
  seed2_ = static_cast<int64_t>(kernel_ptr->get_seed2());
  return true;
}
// 计算算子运行过程中所需内存空间的大小，注意单位为`byte`
int RandomPoissonGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  // 检查所有的输入维度是否合法
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  // 在计算本轮计算所需内存空间时，首先需要将之前产生的脏数据清空
  ResetResource();
  // 计算输入输出的元素的个数
  std::vector<int64_t> shape_shape = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                          inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> rate_shape = std::vector<int64_t>(inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(),
                                                         inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> output_shape = std::vector<int64_t>(outputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                           outputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  int64_t shape_elements = std::accumulate(shape_shape.begin(), shape_shape.end(), 1, std::multiplies<int64_t>());
  rate_elements_ = std::accumulate(rate_shape.begin(), rate_shape.end(), 1, std::multiplies<int64_t>());
  output_elements_ = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
  if (output_elements_ == 0) {
    is_null_input_ = true;
  }
  // 将计算结果储存到数组中
  input_size_list_.emplace_back(shape_elements * unit_shape_size_);
  input_size_list_.emplace_back(rate_elements_ * unit_rate_size_);
  output_size_list_.emplace_back(output_elements_ * unit_output_size_);
  workspace_size_list_.push_back(output_elements_ * sizeof(curandState));
  return KRET_OK;
}
// 算子正式执行入口函数
template <typename R, typename T>
bool RandomPoissonGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  // 获取输入输出的Device端地址
  R *rate_addr = GetDeviceAddress<R>(inputs, 1);
  T *output = GetDeviceAddress<T>(outputs, 0);
  curandState *devStates = nullptr;
  void *workspace_addr = GetDeviceAddress<void *>(workspace, 0);
  devStates = reinterpret_cast<curandState *>(workspace_addr);
  // CUDA核函数
  RandomPoisson(seed_, seed2_, devStates, rate_addr, rate_elements_, output, output_elements_,
                reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}
// 注册算子支持的参数类型，由于参数排列组合类型比较多，因此将实际注册的函数抽离成一个宏定义，使得注册列表看起来更简洁
const std::vector<std::pair<KernelAttr, KernelRunFunc>> &RandomPoissonGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    ADD_KERNEL(Int32, Float16, Float16, half, half),     ADD_KERNEL(Int32, Float16, Float32, half, float),
    ADD_KERNEL(Int32, Float16, Float64, half, double),   ADD_KERNEL(Int32, Float16, Int32, half, int),
    ADD_KERNEL(Int32, Float16, Int64, half, int64_t),

    ADD_KERNEL(Int32, Float32, Float16, float, half),    ADD_KERNEL(Int32, Float32, Float32, float, float),
    ADD_KERNEL(Int32, Float32, Float64, float, double),  ADD_KERNEL(Int32, Float32, Int32, float, int),
    ADD_KERNEL(Int32, Float32, Int64, float, int64_t),

    ADD_KERNEL(Int32, Float64, Float16, double, half),   ADD_KERNEL(Int32, Float64, Float32, double, float),
    ADD_KERNEL(Int32, Float64, Float64, double, double), ADD_KERNEL(Int32, Float64, Int32, double, int),
    ADD_KERNEL(Int32, Float64, Int64, double, int64_t),

    ADD_KERNEL(Int32, Int32, Float16, int, half),        ADD_KERNEL(Int32, Int32, Float32, int, float),
    ADD_KERNEL(Int32, Int32, Float64, int, double),      ADD_KERNEL(Int32, Int32, Int32, int, int),
    ADD_KERNEL(Int32, Int32, Int64, int, int64_t),

    ADD_KERNEL(Int32, Int64, Float16, int64_t, half),    ADD_KERNEL(Int32, Int64, Float32, int64_t, float),
    ADD_KERNEL(Int32, Int64, Float64, int64_t, double),  ADD_KERNEL(Int32, Int64, Int32, int64_t, int),
    ADD_KERNEL(Int32, Int64, Int64, int64_t, int64_t),

    ADD_KERNEL(Int64, Float16, Float16, half, half),     ADD_KERNEL(Int64, Float16, Float32, half, float),
    ADD_KERNEL(Int64, Float16, Float64, half, double),   ADD_KERNEL(Int64, Float16, Int32, half, int),
    ADD_KERNEL(Int64, Float16, Int64, half, int64_t),

    ADD_KERNEL(Int64, Float32, Float16, float, half),    ADD_KERNEL(Int64, Float32, Float32, float, float),
    ADD_KERNEL(Int64, Float32, Float64, float, double),  ADD_KERNEL(Int64, Float32, Int32, float, int),
    ADD_KERNEL(Int64, Float32, Int64, float, int64_t),

    ADD_KERNEL(Int64, Float64, Float16, double, half),   ADD_KERNEL(Int64, Float64, Float32, double, float),
    ADD_KERNEL(Int64, Float64, Float64, double, double), ADD_KERNEL(Int64, Float64, Int32, double, int),
    ADD_KERNEL(Int64, Float64, Int64, double, int64_t),

    ADD_KERNEL(Int64, Int32, Float16, int, half),        ADD_KERNEL(Int64, Int32, Float32, int, float),
    ADD_KERNEL(Int64, Int32, Float64, int, double),      ADD_KERNEL(Int64, Int32, Int32, int, int),
    ADD_KERNEL(Int64, Int32, Int64, int, int64_t),

    ADD_KERNEL(Int64, Int64, Float16, int64_t, half),    ADD_KERNEL(Int64, Int64, Float32, int64_t, float),
    ADD_KERNEL(Int64, Int64, Float64, int64_t, double),  ADD_KERNEL(Int64, Int64, Int32, int64_t, int),
    ADD_KERNEL(Int64, Int64, Int64, int64_t, int64_t)};
  return func_list;
}
// 注册随机泊松算子的GPU Kernel
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, RandomPoisson, RandomPoissonGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
```

## CUDA核函数实现

利用GPU并行计算能力，在计算量大的情况下，算子的加速效果十分明显。CUDA核函数的实现也分为函数定义和函数实现，分别写在文件`mindspore/ccsrc/plugin/device/gpu/kernel/cuda_impl/cuda_ops/random_op_impl.cuh`和文件`mindspore/ccsrc/plugin/device/gpu/kernel/cuda_impl/cuda_ops/random_op_impl.cu`中。

```c++
// 定义CUDA核函数头文件
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RANDOM_OP_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RANDOM_OP_IMPL_CUH_
#include <curand_kernel.h>
#include <random>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
// 随机泊松算子CUDA核函数定义
template <typename R, typename T>
CUDA_LIB_EXPORT void RandomPoisson(int seed, int seed2, curandState *globalState,
                                   R *rate, int64_t rate_size, T *output, size_t count,
                                   cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RANDOM_OP_IMPL_CUH_
```

```c++
#include "random_op_impl.cuh"
#include "include/cuda_fp16.h"
// CUDA核函数实现，注意只有`__global__`字段的函数才会在GPU上执行
template <typename R, typename T>
__global__ void RandomPoissonKernel(int seed, curandState *globalState, R *rate, int rate_size, T *output,
                                    size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    // 使用curand库的接口，初始化curandstate
    curand_init(seed, i, 0, &globalState[i]);
    // 将结果的顺序与`rate`的顺序对齐
    auto j = i % rate_size;
    // 使用curand库的泊松接口生成随机泊松值
    output[i] = (T)curand_poisson(&globalState[i], rate[j]);
  }
  return;
}
// CUDA核函数入口
template <typename R, typename T>
void RandomPoisson(int seed, int seed2, curandState *globalState, R *rate, int64_t rate_size, T *output, size_t count,
                   cudaStream_t cuda_stream) {
  // 使用两个随机种子seed、seed2对random_device类进行初始化
  int RNG_seed = 0;
  std::random_device rd;
  if (seed2 != 0) {
    RNG_seed = seed2;
  } else if (seed != 0) {
    RNG_seed = seed;
  } else {
    RNG_seed = static_cast<int>(rd());
  }
  // 调用CUDA核函数
  RandomPoissonKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(RNG_seed, globalState, rate, rate_size,
                                                                          output, count);
  return;
}
// 注册核函数支持的数据类型，由于参数类型组合比较多，因此使用宏定义抽离
#define ADD_RANDOM_POISSON(rate_type, output_type) \
  template CUDA_LIB_EXPORT void RandomPoisson<rate_type, output_type>(int seed, int seed2, curandState *globalState, \
                                                                      rate_type *rate, int64_t rate_size, \
                                                                      output_type *output, size_t count, \
                                                                      cudaStream_t cuda_stream);

ADD_RANDOM_POISSON(half, half)
ADD_RANDOM_POISSON(half, float)
ADD_RANDOM_POISSON(half, double)
ADD_RANDOM_POISSON(half, int)
ADD_RANDOM_POISSON(half, int64_t)

ADD_RANDOM_POISSON(float, half)
ADD_RANDOM_POISSON(float, float)
ADD_RANDOM_POISSON(float, double)
ADD_RANDOM_POISSON(float, int)
ADD_RANDOM_POISSON(float, int64_t)

ADD_RANDOM_POISSON(double, half)
ADD_RANDOM_POISSON(double, float)
ADD_RANDOM_POISSON(double, double)
ADD_RANDOM_POISSON(double, int)
ADD_RANDOM_POISSON(double, int64_t)

ADD_RANDOM_POISSON(int, half)
ADD_RANDOM_POISSON(int, float)
ADD_RANDOM_POISSON(int, double)
ADD_RANDOM_POISSON(int, int)
ADD_RANDOM_POISSON(int, int64_t)

ADD_RANDOM_POISSON(int64_t, half)
ADD_RANDOM_POISSON(int64_t, float)
ADD_RANDOM_POISSON(int64_t, double)
ADD_RANDOM_POISSON(int64_t, int)
ADD_RANDOM_POISSON(int64_t, int64_t)
```

# 算子执行流程图

对算子源码分析完成后，接下来用一副流程图来描述随机泊松算子的完整执行过程。

![执行流程图](/api/attachments/396221)

# 算子测试

对算子计算结果进行测试，测试覆盖了算子支持的所有数据类型，脚本如下：

```python
import mindspore as ms
from mindspore import Tensor
from mindspore import ops
import numpy as np
import time

start_time = time.time()
ms.set_context(device_target="GPU")

int_type = [ms.int32, ms.int64]
float_type = [ms.int32, ms.int64, ms.float16, ms.float32, ms.float64]

for x in int_type:
    for y in float_type:
        for z in float_type:
            shape = Tensor(np.array([2, 4]), x)
            rate = Tensor(np.array([5, 50, 500]), y)
            seed = 10
            seed2 = 20
            random_poisson = ops.operations.random_ops.RandomPoisson(seed=seed, seed2=seed2, dtype=z)
            output = random_poisson(shape, rate)
            print('=====Devided line=====')
            print(output)
            print(output.shape)
            print(output.dtype)

end_time = time.time()
print('Process time(s): ', end_time - start_time)
```

算子测试结果的部分截图如下：

![测试结果](/api/attachments/396222)
