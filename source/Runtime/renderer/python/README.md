# Ruzino Python Render Graph - 渲染图Python接口

## 概述 (Overview)

这个模块为Ruzino渲染系统提供了Python接口,类似于几何节点的`headless_geom_node_executor`。它允许你通过Python脚本构建和执行渲染图,无需GUI界面。

This module provides a Python interface for the Ruzino rendering system, similar to the `headless_geom_node_executor` for geometry nodes. It allows you to build and execute render graphs through Python scripts without requiring a GUI.

## 主要特性 (Key Features)

### 1. 渲染图API (Render Graph API)
- **RuzinoRenderGraph**: 扩展自`RuzinoGraph`,专门用于渲染
- 加载USD场景文件
- 设置渲染参数(分辨率、SPP等)
- 执行渲染图

### 2. CUDA内存转换节点 (CUDA Memory Conversion Nodes)
- **nvrhi_to_cuda**: 将NVRHI纹理转换为CUDA线性缓冲区
- **cuda_to_nvrhi**: 将CUDA线性缓冲区转换为NVRHI纹理
- 支持DLPack协议,可与PyTorch/CuPy等无缝互操作
- 直接在GPU上操作,无需CPU中转

### 3. Python绑定 (Python Bindings)
- **rzrenderer_py**: 渲染器Python绑定模块
- RenderGlobalPayload暴露
- CUDA缓冲区支持DLPack导出
- USD Stage操作
- RHI系统初始化

## 文件结构 (File Structure)

```
source/Runtime/renderer/
├── nodes/
│   └── conversion/
│       ├── nvrhi_to_cuda.cpp        # NVRHI -> CUDA转换节点
│       └── cuda_to_nvrhi.cpp        # CUDA -> NVRHI转换节点
├── python/
│   ├── renderer.cpp                 # Python绑定实现 (rzrenderer_py)
│   ├── ruzino_render_graph.py      # 渲染图Python API
│   ├── headless_render_executor.py # 无头渲染执行器
│   ├── example_render.py           # 示例脚本
│   └── README.md                    # 本文档
└── tests/
    ├── test_render_graph.py         # 高层API测试
    ├── test_path_tracing.py         # 路径追踪测试
    └── test_render_direct.py        # 直接C++绑定测试
```

## 使用方法 (Usage)

### 直接使用节点系统 (Direct Node System Usage)

最简单直接的方式是使用底层节点系统API:

```python
import nodes_system
import stage_py

# 创建节点系统
system = nodes_system.create_dynamic_loading_system()
system.load_configuration("render_nodes.json")
system.init()

# 加载USD场景
stage = stage_py.Stage("scene.usdc")
payload = stage_py.create_payload_from_stage(stage)
meta_payload = stage_py.create_meta_any_from_payload(payload)
system.set_global_params_any(meta_payload)

# 加载并执行渲染图
with open("render_graph.json") as f:
    graph_json = f.read()

node_tree = system.get_node_tree()
node_tree.deserialize(graph_json)
system.execute(mark_all_required=True)

# 清理
system.finalize()
```

### 运行测试 (Running Tests)

```bash
cd build/bin

# 直接测试 (推荐 - 最简单)
python ../../source/Runtime/renderer/tests/test_render_direct.py

# 使用shader_ball场景和完整路径追踪管线
# 这会使用 Assets/render_nodes_save.json 和 Assets/shader_ball.usdc
```

## 构建说明 (Build Instructions)

### 前置条件 (Prerequisites)

1. **PyTorch** (可选,用于转换节点):
```bash
pip install torch torchvision
```

2. **其他Python依赖**:
```bash
pip install numpy pillow  # 用于图像保存
```

### CMake配置 (CMake Configuration)

转换节点会在检测到PyTorch时自动启用。确保在CMake配置时定义了`USTC_CG_WITH_TORCH`:

```bash
cmake -B build -DUSTC_CG_WITH_TORCH=ON
cmake --build build
```

### 重新构建 (Rebuilding)

当你修改C++代码后:

```bash
# 如果添加了新文件,需要重新运行CMake
cmake -B build

# 构建
cmake --build build --target hd_USTC_CG
cmake --build build --target render_nodes
```

当你添加新节点后,需要重新生成JSON配置:

```bash
cmake --build build --target render_nodes_json_target
```

## 测试 (Testing)

运行测试套件:

```bash
cd build
python ../source/Runtime/renderer/tests/test_render_graph.py
```

或使用pytest:

```bash
pytest source/Runtime/renderer/tests/test_render_graph.py -v
```

## API参考 (API Reference)

### RuzinoRenderGraph类

继承自`RuzinoGraph`,增加了渲染特定功能:

#### 方法 (Methods)

- `loadUSDStage(usd_path: str)`: 加载USD场景
- `setRenderSettings(width, height, spp)`: 设置渲染参数
- `initRHI()`: 初始化渲染硬件接口
- `getOutputAsTensor(node, socket_name)`: 获取输出为Tensor
- `saveOutputImage(node, socket_name, output_path, gamma)`: 保存输出图像

#### 属性 (Properties)

- `render_settings`: 当前渲染设置字典
- `usd_stage`: 加载的USD舞台

## 注意事项 (Notes)

1. **Executor区别**: 渲染图使用`EagerNodeTreeExecutorRender`,而几何图使用标准的Executor。这是因为渲染需要特殊的资源管理和调度。

2. **内存管理**: 转换节点在GPU和CPU之间传输数据。大型纹理可能消耗大量内存。

3. **CUDA同步**: 转换节点会自动处理CUDA同步,但在批量操作时要注意性能。

4. **类型注册**: 转换节点需要正确注册类型系统。确保`USTC_CG_WITH_TORCH`在编译时定义。

## 故障排除 (Troubleshooting)

### 问题: 找不到renderer_py模块

**解决**: 确保Python能找到编译的模块:
```python
import sys
sys.path.insert(0, 'path/to/build/bin')
```

### 问题: 转换节点不可用

**解决**: 检查PyTorch是否正确安装,以及CMake配置中是否启用了`USTC_CG_WITH_TORCH`。

### 问题: CUDA错误

**解决**: 确保PyTorch和NVRHI使用相同的CUDA版本。

## 示例场景 (Example Scenarios)

### 场景1: 批量渲染

```python
scenes = ["scene1.usdc", "scene2.usdc", "scene3.usdc"]
for scene in scenes:
    quick_render(
        usd_path=scene,
        json_script="render_config.json",
        output_image=f"output_{scene}.png",
        width=1920,
        height=1080,
        spp=16
    )
```

### 场景2: 渲染后处理

```python
g = RuzinoRenderGraph("PostProcess")
g.initRHI()
g.loadConfiguration("render_nodes.json")
g.loadUSDStage("scene.usdc")

# 渲染
path_trace = g.createNode("path_tracing")
g.markOutput(path_trace, "Output")
g.execute()

# 获取结果
result = g.getOutputAsTensor(path_trace, "Output")

# 用PyTorch做后处理
import torch
denoised = my_denoise_function(result)

# 保存
from torchvision.utils import save_image
save_image(denoised, "final_output.png")
```

## 进一步开发 (Further Development)

### 添加新转换节点

1. 在`nodes/conversion/`创建`.cpp`文件
2. 实现`declare`和`exec`函数
3. 标记为转换节点: `ntype.is_conversion_node = true`
4. 重新构建

### 扩展Python API

编辑`ruzino_render_graph.py`添加新方法。保持API简洁和链式调用风格。

## 许可证 (License)

与Ruzino项目主体相同。

## 贡献 (Contributing)

欢迎提交问题和Pull Request到主仓库。
