# run this script after bash ```source /usr/local/Ascend/ascend-toolkit/set_env.sh```
import torch
import torch_npu

# 检查 NPU 是否可用
npu_available = torch.npu.is_available()
print(f"NPU is available: {npu_available}")


# 在 NPU（或 CPU）上创建张量
x = torch.randn([1000, 1000]).npu

# 持续矩阵乘，加大 NPU 负载
while True:
    x = torch.matmul(x, x)
