import time
import torch

start = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = torch.rand(1000, 1000).to(device)
result = tensor.sum()
print(f'Device used: {device}')
print('Operation successful')

end = time.time()
print(f'Benchmark time: {end - start} seconds')
