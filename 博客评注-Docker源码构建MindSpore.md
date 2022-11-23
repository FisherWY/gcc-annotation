# docker 源码构建 mindspore

本文记录了 docker 源码构建 mindspore 的过程。

大致分为如下几个步骤：

第一，获取 mindspore 源码

第二，拉取 mindspore 构建的镜像

第三，使用镜像构建源码

第四，安装 mindspore，检查是否正确构建。

# 获取 mindspore 源码

mindspore 在 gitee 上面开放了源代码，我们可以非常快速的拉取到 mindspore 的源码，使用下面命令，拉取 mindspore 源码。

```
git clone https://gitee.com/mindspore/mindspore.git
```

# 拉取 mindspore 构建的镜像

mindspore 在 docker hub 上面提供了构建的源码的镜像。我们可以搜索 docker hub，可以发现 mindspore 提供了两个镜像，一个是用于构建 cpu 版本的，一个是用于构建 gpu 版本的。其中，每个版本由分别提供了 `x.y.z` 预安装 mindspore 的版本，`devel` 开发环境，`runtime` 运行环境。

![mindspore-docker](/api/attachments/387958 "mindspore-docker")

为了构建 mindspore，我们使用下面命令拉取 `devel` 版本的 mindspore 编译环境镜像。

```
docker pull mindspore/mindspore-gpu:devel
```

# 使用镜像构建源码

拉取好了镜像之后，我们可以使用下面的命令启动一个容器。注意，运行这条命令的目录，是在 clone 下来的 mindspore 的源码目录里面。

```
docker run --runtime=nvidia --network=host --shm-size=8g -v$(pwd):$(pwd) -w$(pwd)  -it pull mindspore/mindspore-gpu:devel
```

进入了容器之后，我们可以使用下面命令启动 mindspore 提供的构建脚本。下面的 `-j64` 依据机器配置来，默认是 `j8`。之后就会开始构建 mindspore，整个过程可能需要 1 个小时，耐心等待这个大型 C++ 项目构建吧。

```
bash build.sh -e gpu -S on -j64
```

# 安装 mindspore

编译的 mindspore 会在 build 目录里面产生一个 pip 包，我们可以直接使用 pip 安装。虽然官方推荐在 devel 环境构建，在 runtime 环境中运行，但是我就不！反正能跑。

```
pip3 install build/package/mindspore_gpu-1.8.0-cp37-cp37m-linux_x86_64.whl
```

安装好了之后，需要运行一下检查程序，先进入 python3 的交互界面。之后 `import mindspore` 即可。可以看到 mindspore 已经顺利构建。

```
Python 3.7.5 (default, Oct 28 2021, 07:07:57) 
[GCC 7.5.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import mindspore
>>> mindspore.run_check()
MindSpore version:  1.8.0
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

# 续

我们在 mindspore 主目录下面的 CMakeLists.txt 中，第 54 行，我们可以看到如下代码：

```
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
```

这行代码的作用是导出一个 `compile_commands.json`，它有什么作用呢？

如果随便点开看看，你会发现里面都是编译的命令，给出了 CMake 对每个源文件的编译命令，编译参数。借助这个文件，我们可以使用 clangd 等程序，搭建 mindspore 的开发环境，比如使用 VSCode + clangd，我们可以轻松拥有代码提示，代码跳转等功能。

读到这里，如果您还想知道怎么搭建开发环境，拥有代码提示、代码跳转、代码补全等功能，可以给我留言，后面可以写一篇相关的文章。
