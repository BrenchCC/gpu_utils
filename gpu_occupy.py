import os
import torch
import multiprocessing
import time

# 定义要在单个 GPU 上执行的函数
def run_on_gpu(gpu_id):
    torch.cuda.set_device(gpu_id)
    while True:
        with torch.inference_mode():
            # 这里的矩阵大小可以根据需要调整以改变负载
            a = torch.randn(10000, 10000, device=gpu_id)
            b = torch.randn(10000, 10000, device=gpu_id)
            c = torch.matmul(a, b)
            d = torch.matmul(a, c)
            e = torch.matmul(d, b)

def monitor_gpus(num_gpus):
    try:
        import pynvml
        pynvml.nvmlInit()
    except ImportError:
        print("pynvml not installed. Please install it with 'pip install pynvml'.")
        return

    while True:
        try:
            os.system('clear' if os.name == 'posix' else 'cls')
            print(f"{'GPU ID':<10} | {'Memory Usage (%)':<20} | {'GPU Utilization (%)':<20}")
            print("-" * 55)
            for i in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                mem_usage_pct = (mem_info.used / mem_info.total) * 100
                gpu_util_pct = utilization.gpu
                
                print(f"{i:<10} | {mem_usage_pct:<20.2f} | {gpu_util_pct:<20}")
            time.sleep(1)
        except Exception as e:
            print(f"Monitoring error: {e}")
            break

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found. Exiting.")
        exit(0)
    
    processes = []
    
    # 启动监控进程
    monitor_process = multiprocessing.Process(target=monitor_gpus, args=(num_gpus,))
    monitor_process.daemon = True
    monitor_process.start()
    
    # 启动 GPU 负载进程
    print(f"Starting load on {num_gpus} GPUs...")
    for i in range(num_gpus):
        p = multiprocessing.Process(target=run_on_gpu, args=(i,))
        p.start()
        processes.append(p)
        
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nStopping processes...")
        for p in processes:
            p.terminate()
            p.join()
        print("Done.")