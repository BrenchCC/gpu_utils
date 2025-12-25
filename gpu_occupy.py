import os
import torch
import torch_npu

# 定义要在单个 NPU 上执行的函数
def run_on_npu(npu_id):
    # 设置当前 NPU 设备
    torch.npu.set_device(npu_id)
    
    while True:
        with torch.inference_mode():
            # 在 NPU 上创建张量
            a = torch.randn(50, 50, device=f'npu:{npu_id}')
            b = torch.randn(50, 50, device=f'npu:{npu_id}')
            c = torch.matmul(a, b)
            d = torch.matmul(a, c)
            e = torch.matmul(d, b)
            #print(f"NPU {npu_id} result: {c}")

if __name__ == "__main__":
    import multiprocessing
    
    # 检查是否有可用的 NPU
    if not torch.npu.is_available():
        print("NPU is not available")
        exit(1)
    
    # 获取 NPU 数量
    num_npus = torch.npu.device_count()
    print(f"Found {num_npus} NPU(s)")
    
    processes = []
    for i in range(num_npus):
        p = multiprocessing.Process(target=run_on_npu, args=(i,))
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
