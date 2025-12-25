import os
import torch


# 定义要在单个 GPU 上执行的函数
def run_on_gpu(gpu_id):
    torch.cuda.set_device(gpu_id)
    while True:
        with torch.inference_mode():
            a = torch.randn(10000, 10000, device=gpu_id)
            b = torch.randn(10000, 10000, device=gpu_id)
            c = torch.matmul(a, b)
            d = torch.matmul(a, c)
            e = torch.matmul(d, b)
            #print(f"GPU {gpu_id} result: {c}")

if __name__ == "__main__":
    import multiprocessing
    num_gpus = torch.cuda.device_count()
    processes = []
    for i in range(num_gpus):
        p = multiprocessing.Process(target=run_on_gpu, args=(i,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
