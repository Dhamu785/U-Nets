# %% import base libs and calculate memory
import psutil

def resources_cpu(memory, memory_used=None):
    total_memory = memory.total
    available_memory = memory.available
    used_memory = total_memory - available_memory

    
    print("RAM :")
    # print(f"Total memory: {total_memory / (1024**3):.2f} GB")
    print(f"    Used memory: {used_memory / (1024**3):.2f} GB")
    print(f"    Available memory: {available_memory / (1024**3):.2f} GB")
    print(f"    Memory usage percentage: {memory.percent}%")
    if memory_used:
        print(f"Total memory used now = {(used_memory-memory_used)/(1024**3):.2f} GB")
    print("----"*20)
    return used_memory
memory = psutil.virtual_memory()
memory_used = resources_cpu(memory)
# %% import AI libaries
import torch as t
import gc
from torchvision import transforms
from PIL import Image
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'

# %% get memory usages after loading ml libaries
def resources_gpu(memory, memory_used=None):
    total_memory = memory.total
    available_memory = memory.available
    used_memory = total_memory - available_memory

    
    print("RAM")
    # print(f"Total memory: {total_memory / (1024**3):.2f} GB")
    print(f"    Used memory: {used_memory / (1024**3):.2f} GB")
    print(f"    Available memory: {available_memory / (1024**3):.2f} GB")
    print(f"    Memory usage percentage: {memory.percent}%")
    if memory_used:
        print(f"Total RAM used now = {(used_memory-memory_used)/(1024**3):.2f} GB")
    print("GPU")
    print(f"    GPU Memory = {t.cuda.memory_allocated()/(1024*1024):.2f} MB")

    print("----"*20)
    return used_memory
memory = psutil.virtual_memory()
memory_used = resources_gpu(memory, memory_used)
# %% Image quality enhancement
print("BEFORE LOADING THE MODEL")
memory = psutil.virtual_memory()
memory_used = resources_gpu(memory, memory_used)

model = t.load('Image_enhancement-em.pt')
print("AFTER LOADING THE MODEL")
memory = psutil.virtual_memory()
memory_used = resources_gpu(memory, memory_used)

transforms_pipe = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ])
img = Image.open('D:\\Image quality enhancement\\data\\X\\18518015.jpeg').convert('RGB')
img_transformed = transforms_pipe(img)
img = img_transformed.unsqueeze(0).to(DEVICE)
with t.inference_mode():
    pred = model(img)
print("AFTER PREDICTIONS")
memory = psutil.virtual_memory()
memory_used = resources_gpu(memory, memory_used)

del img, img_transformed, pred
gc.collect()
t.cuda.empty_cache()
print("AFTER DELETING VARIABLES")
memory = psutil.virtual_memory()
memory_used = resources_gpu(memory, memory_used)

img = Image.open('D:\\Image quality enhancement\\data\\X\\18518015.jpeg').convert('RGB')
img_transformed = transforms_pipe(img)
img = img_transformed.unsqueeze(0).to(DEVICE)
with t.inference_mode():
    pred = model(img)
print("AFTER SECOND PREDICTION")
memory = psutil.virtual_memory()
memory_used = resources_gpu(memory, memory_used)