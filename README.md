# gpu_utils

一个简单的设备占用工具集合，用于快速拉高 GPU 或 NPU 负载，方便做资源占用、调度测试和环境验证。

## 文件说明

- `gpu_occupy.py`: 对当前机器上的所有 GPU 持续执行矩阵计算，并实时显示显存占用率和 GPU 利用率。
- `npu_guarder.py`: 对当前机器上的所有 NPU 持续执行矩阵计算。

## 依赖

### GPU

- `torch`
- `pynvml`

安装示例：

```bash
pip install torch pynvml
```

### NPU

- `torch`
- `torch_npu`
- 已正确安装并配置 Ascend 相关环境

运行前通常需要先执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 使用方法

启动 GPU 占用：

```bash
python gpu_occupy.py
```

启动 NPU 占用：

```bash
python npu_guarder.py
```

## 说明

- 脚本会默认使用当前机器上的全部可见设备。
- 这些脚本会持续运行，直到手动停止。
- 停止运行可使用 `Ctrl+C`。
- 运行期间可能会占用大量计算资源和显存，请谨慎在生产环境使用。
