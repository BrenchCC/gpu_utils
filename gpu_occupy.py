import torch

cuda_ava = torch.cuda.is_available()

print(f"CUDA is available: {cuda_ava}")

x = torch.randn([1000, 1000]).cuda()

while True:
    torch.matmul(x,x)
