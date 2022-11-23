# 一个算子在 MindSpore 框架中的执行流程

本文分析了一个算子在 MindSpore 框架中的执行流程。MindSpore 中设计了 Primitive 对算子进行了封装和抽象，一般来说封装和抽象是出于差异，这种差异来自于底层执行设备的差异，比如有 CPU，GPU，Ascend 等执行设备，每种执行设备上的计算逻辑，内存分配，通信逻辑各不相同。“没有什么问题是加一层抽象不能解决的”，如果有，咱们加两层哈哈。本文着重分析一个算子在 MindSpore 框架中的执行流程，对 Primitive 的设计论述相对较少，但是通过观察一个算子在框架中的执行流程，我们可以形象的感知到 Primitive 的作用。后续将会写一篇文章分析 Primitive 的设计。

原文发布在 GitLink 上的 MindSpore 评注解读上面，可以的话点一点链接！

链接：https://forum.gitlink.org.cn/forums/7330/detail

# Python ReLU 代码

首先写一个最简单的算子，计算一个向量的 ReLU。

```python
import mindspore
import mindspore.ops as P
from mindspore import Tensor
from mindspore import dtype as mstype
import numpy as np

input_x = Tensor(np.random.randn(2,3), mstype.float32)
relu = P.ReLU()
output_x = relu(input_x)
print(input_x)
print(output_x)
```

输出如下，大于 0 的部分保留，小于 0 的部分设为 0：

```
[[ 1.5206318  -0.35908994 -0.54122275]
 [ 0.32850873 -0.6513135  -2.8261368 ]]
[[1.5206318  0.         0.        ]
 [0.32850873 0.         0.        ]]
```

# Python 端源码分析

ReLU 继承了 Primitive，Primitive 又继承自 C++ 导出的 `Primitive_`。

调用 ReLU 的背后，实际上执行的是 `_run_op` 这个函数

```python
# mindspore/python/mindspore/ops/primitive.py
def __call__(self, *args):
    should_elim, output = self.check_elim(*args)
    for arg in args:
        if isinstance(arg, Parameter) and arg.has_init:
            arg.init_data()
    if should_elim:
        return output
    return _run_op(self, self.name, args)
```

`_run_op` 调用了 C++ 导出的 `real_run_op`

```python
# mindspore/python/mindspore/ops/primitive.py
@_wrap_func
def _run_op(obj, op_name, args):
    """Single op execution function supported by ge in PyNative mode."""
    output = real_run_op(obj, op_name, args)
    return output
```

由此我们进入到 C++ 端的实现中。

# C++ 端源码分析

首先看看 pybind11 导出的函数。

```cpp
// mindspore/ccsrc/pipeline/jit/init.cc
(void)m.def("real_run_op", &mindspore::pynative::RealRunOp, "Run op pynatively.");
```

RealRunOp 内部实现如下，先构造 info，然后再执行。info 里面会保存参数，保存 Python 端的 Primitive 对象。

```cpp
// mindspore/ccsrc/pipeline/pynative/pynative_execute.cc
py::object RealRunOp(const py::args &args) {
  CheckPyNativeContext();
  const auto &executor = PynativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor);
  OpExecInfoPtr op_exec_info = executor->forward_executor()->GenerateOpExecInfo(args);
  MS_EXCEPTION_IF_NULL(op_exec_info);
  py::object ret = py::none();
  PynativeExecutorTry(executor->forward_executor()->RunOpS, &ret, op_exec_info);
  return ret;
}
```

传递到 ForwardExecutor 内部的 RunOpInner。首先执行检查，判断参数正确性，判断算子是否存在，特殊处理混合精度算子。因为是 PyNative，所以是动态图，也就是执行的时候构图，可以从下面看到 `ConstructForwardGraph`，最后通过 `GetOpOutput` 执行算子。

```cpp
std::function<void(py::object *, const OpExecInfoPtr &)> RunOpS = [this](auto &&PH1, auto &&PH2) {
  RunOpInner(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2));
};

void ForwardExecutor::RunOpInner(py::object *ret, const OpExecInfoPtr &op_exec_info) {
  MS_EXCEPTION_IF_NULL(ret);
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_LOG(DEBUG) << "RunOp name: " << op_exec_info->op_name;
  if (kSummaryOperators.count(op_exec_info->op_name)) {
    MS_LOG(DEBUG) << "PyNative not support Operator " << op_exec_info->op_name;
    return;
  }
  if (op_exec_info->op_name == prim::kPrimMixedPrecisionCast->name()) {
    RunMixedPrecisionCastOp(op_exec_info, ret);
    return;
  }

  // 1.Set cast for inputs
  SetCastForInputs(op_exec_info);
  // 2.Construct graph, first step abs will update by node
  auto cnode = ConstructForwardGraph(op_exec_info);
  // 3.Get inputs abstract
  abstract::AbstractBasePtrList args_spec_list;
  GetInputsArgsSpec(op_exec_info, &args_spec_list);
  // 4.Get output abstract
  bool prim_cache_hit = false;
  GetOpOutputAbstract(op_exec_info, args_spec_list, &prim_cache_hit);
  // 5.Get output
  GetOpOutput(op_exec_info, args_spec_list, cnode, prim_cache_hit, ret);
}
```

GetOpOutput 内部还要处理反向梯度，需要将算子的信息记录下来。最终执行到 RunOpWithInitBackendPolicy 里面，使用指定的后端来运行算子。后端指的是运行引擎，比如 vm，ge，tsd 等，不同的运行引擎应该有不同的优化策略。

```cpp
void ForwardExecutor::GetOpOutput(const OpExecInfoPtr &op_exec_info,
                                  const abstract::AbstractBasePtrList &args_spec_list, const CNodePtr &cnode,
                                  bool prim_cache_hit, py::object *ret) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  const auto &prim = op_exec_info->py_primitive;
  MS_EXCEPTION_IF_NULL(prim);
  // Infer output value by constant folding
  MS_EXCEPTION_IF_NULL(ret);
  py::dict output = abstract::ConvertAbstractToPython(op_exec_info->abstract, true);
  if (!output[ATTR_VALUE].is_none()) {
    *ret = output[ATTR_VALUE];
    grad()->RecordGradOpInfo(op_exec_info);
    MS_LOG(DEBUG) << "Get output by constant folding, output is " << py::str(*ret);
    return;
  } else if (prim->is_const_prim()) {
    *ret = py::cast("");
    grad()->RecordGradOpInfo(op_exec_info);
    MS_LOG(DEBUG) << "Get const prim";
    return;
  }

  // Add output abstract info into cache, the const value needs to infer evert step
  if (grad()->enable_op_cache() && !prim_cache_hit && !IsDynamicShape(op_exec_info)) {
    AbsCacheKey key{prim->name(), prim->Hash(), prim->attrs()};
    auto &out = prim_abs_list_[key];
    out[args_spec_list].abs = op_exec_info->abstract;
    out[args_spec_list].attrs = prim->evaluate_added_attrs();
  }

  // Run op with selected backend, nop is no need run backend
  ValuePtr out_real_value = nullptr;
  if (op_exec_info->is_nop_prim) {
    DoNopOutput(op_exec_info, &out_real_value);
    *ret = BaseRefToPyData(out_real_value);
  } else {
    auto result = RunOpWithInitBackendPolicy(op_exec_info);
    py::object out_real = result;
    if (result.size() == 1 && op_exec_info->abstract != nullptr &&
        !op_exec_info->abstract->isa<abstract::AbstractSequence>()) {
      out_real = result[0];
    }
    // Get output value
    if (grad()->grad_flag()) {
      out_real_value = PyObjToValue(out_real);
    }
    *ret = out_real;
  }

  if (grad()->need_construct_graph() && !grad()->in_cell_with_custom_bprop_()) {
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &obj_id = GetId(*ret);
    cnode->set_abstract(op_exec_info->abstract);
    node_abs_map_[obj_id] = op_exec_info->abstract;
    grad()->SaveOutputNodeMap(obj_id, *ret, cnode);
    grad()->DoOpGrad(op_exec_info, cnode, out_real_value);
    // Dynamic shape should update to top cell
    if (IsDynamicShape(op_exec_info)) {
      grad()->top_cell()->set_dynamic_shape(true);
    }
  } else {
    node_abs_map_.clear();
  }
  // Record op info for judge whether the construct of cell has been changed
  grad()->RecordGradOpInfo(op_exec_info);
  grad()->UpdateForwardTensorInfoInBpropGraph(op_exec_info->op_info, out_real_value);
}
```

接下来进入到根据策略执行算子的地方。如果跟进去看，RunOpInVM，最后的逻辑是运行了 python 的计算函数。如果要往 C++ 走，应该还要看 RunOpInMs，即在 MindSpore 中执行算子。

```cpp
py::object ForwardExecutor::RunOpWithBackendPolicy(MsBackendPolicy backend_policy, const OpExecInfoPtr &op_exec_info) {
  py::object result;
  if (backend_policy == kMsBackendVmOnly) {
#ifndef ENABLE_TEST
    if (kVmOperators.find(op_exec_info->op_name) != kVmOperators.end()) {
      result = RunOpInVM(op_exec_info);
    } else {
      result = RunOpInMs(op_exec_info);
    }
#else
    result = RunOpInVM(op_exec_info);
#endif
  }

  return result;
}
```

在 RunOpInMs 里面，可以看到运行时 Runtime 又分为了 Ms 和 MindRT 两种，我们暂且看 Ms 这一条分支。先获取一个 Session，然后运行算子。Session 需要通过 `MS_REG_SESSION` 宏来注册一个设备对应的 Session 类。这个类是一个注册工厂，在启动的时候，创建对应的类，注册到工厂当中，后面可以通过 `GetCurrentSession` 运行时获取对应的 Session 类。

```cpp
VectorRef outputs;
if (!enable_mind_rt) {
    auto cur_session = GetCurrentSession(cur_target, device_id);
    MS_EXCEPTION_IF_NULL(cur_session);
    cur_session->RunOp(&op_run_info, &outputs);
} else {
    auto cur_mind_rt_backend = GetMindRtBackend(cur_target, device_id);
    MS_EXCEPTION_IF_NULL(cur_mind_rt_backend);
    mindspore::ScopedLongRunning long_running;
    cur_mind_rt_backend->RunOp(&op_run_info, &outputs);
}

// mindspore/ccsrc/backend/common/session/session_factory.h
#define MS_REG_SESSION(DEVICE_NAME, SESSION_CLASS)                           \
  static const SessionRegistrar g_session_registrar__##DEVICE_NAME##_##_reg( \
    DEVICE_NAME, []() { return std::make_shared<SESSION_CLASS>(); });
```

通过搜索使用了 `MS_REG_SESSION` 宏的地方，我们可以看到有 cpu, gpu, ascend 还有其他 session 类调用了。我们专注于看 cpu 这一条分支。上面分析到了，通过跟踪 CPUSession 的父类 SessionBasic，再看它的类成员 Executor，我们可以看到最后其实调用的是每个 Session 的 RunOpImpl 这个函数。于是我们只要专注看 CPUSession 的 RunOpImpl 即可。

调用 `ProcessInputTensorsForHeterogeneous` 如果数据不在 CPU 上，那么将数据同步到 CPU 上。构建 Op 计算图，创建输出向量，再将输入和输出向量绑定到 `device::cpu::CPUKernelRuntime` 对象上，最后调用 Run 方法执行。

```cpp

void CPUSession::RunOpImpl(const GraphInfo &graph_info, OpRunInfo *op_run_info,
                           std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                           const std::vector<int64_t> &tensors_mask) {
  MS_EXCEPTION_IF_NULL(input_tensors);
  MS_EXCEPTION_IF_NULL(op_run_info);
  ProcessInputTensorsForHeterogeneous("CPU", *input_tensors);
  const auto &kernel_graph = BuildOpImpl(*op_run_info, graph_info, *input_tensors, tensors_mask);
  EraseValueNodeTensor(tensors_mask, input_tensors);

  // Remove reorder after PS feature finish adapting push/pull in auto_monad.
  auto execution_order = kernel_graph->execution_order();
  Reorder(&execution_order);
  kernel_graph->set_execution_order(execution_order);

  // runtime init
  if (!runtime_.Init()) {
    MS_LOG(EXCEPTION) << "Kernel runtime init error.";
  }
  runtime_.AssignKernelGraphAddress(kernel_graph.get());
  std::map<tensor::TensorPtr, session::KernelWithIndex> tensor_to_node;
  runtime_.CreateOutputTensors(kernel_graph.get(), *input_tensors, outputs, &tensor_to_node);
  runtime_.BindInputOutput(kernel_graph.get(), *input_tensors, outputs);

  bool ret = runtime_.Run(*kernel_graph, false);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Run Op failed";
  }
  UpdateDynamicOutputShape(tensor_to_node);
  // update output abstract of dynamic op to op_run_info
  if (op_run_info->output_is_dynamic_shape) {
    UpdateOutputAbstract(kernel_graph, op_run_info);
  }
  SetOutputFlags(*outputs);
  runtime_.RunOpClearMemory(*kernel_graph);
}
```

`device::cpu::CPUKernelRuntime` 是 MindSpore Primitive 设计的一部分，Primitive 通过对 “计算设备” 进行封装和抽象，抽象出了 `KernelRuntime` 算子执行的逻辑就在这里面，我们已经快接近真相了。CPUKernelRuntime 的 Run 方法里面为了处理 Profile，安全，调试等，写了不少条件编译，我们只看最核心的逻辑。前面我们可以看到 `kernel_graph` 已经设置好了执行顺序，因此在这里我们只需要依次获取 kernel，然后逐个执行，调用 Kernel 的 Launch 方法。

```cpp
for (const auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    // akg kernel do not support dynamic shape by now
    kernel::NativeCpuKernelMod *cpu_kernel = nullptr;
    if (session::AnfRuntimeAlgorithm::GetKernelType(kernel) != KernelType::AKG_KERNEL) {
        cpu_kernel = dynamic_cast<kernel::NativeCpuKernelMod *>(kernel_mod);
        MS_EXCEPTION_IF_NULL(cpu_kernel);
    }
    if (common::AnfAlgo::IsDynamicShape(kernel)) {
        AnfAlgo::InferShape(kernel);
        auto args = kernel::GetArgsFromCNode(kernel);
        if (cpu_kernel != nullptr && cpu_kernel->Resize(args->op, args->inputs, args->outputs, args->depend_tensor_map) ==
                                        kernel::KRET_RESIZE_FAILED) {
        MS_LOG(EXCEPTION) << "Node " << kernel->fullname_with_scope() << " Resize failed!";
        }
    }
    std::vector<kernel::AddressPtr> kernel_inputs;
    std::vector<kernel::AddressPtr> kernel_workspaces;
    std::vector<kernel::AddressPtr> kernel_outputs;
    GetRuntimeAddressFromNode(kernel, &kernel_inputs, &kernel_outputs, &kernel_workspaces);
    bool ret = true;
    try {
        ret = kernel_mod->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, 0);
    } catch (std::exception &e) {
        MS_LOG(EXCEPTION) << e.what() << trace::DumpSourceLines(kernel);
    }
    if (!ret) {
        MS_LOG(EXCEPTION) << "Launch kernel failed." << trace::DumpSourceLines(kernel);
    }
    static_cast<CPUMemoryManager *>(mem_manager_.get())->DecreaseAddressRefCount(kernel);
}
```

再往下翻，我们找到 ReLU 的 CPU kernel 实现，其实就是判断是否大于 0，大于 0 的保留，小于 0 的设置为 0。

```cpp
template <typename T>
void Relu(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [in, out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = std::greater<T>()(in[i], 0) ? in[i] : 0;
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReLU,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kReLU); });
```

# 总结

自此，我们分析了一个算子从 Python 前端最终运行到 C++ CPU Kernel 的全流程，其他分支流程大同小异，基本遵循着 “构图，图优化，分配内存，绑定输入输出，运行” 这样一个模式。
