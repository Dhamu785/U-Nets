import psutil

def resources_cpu(memory, memory_used=None):
    total_memory = memory.total
    available_memory = memory.available
    used_memory = total_memory - available_memory

    print("RAM :")
    print(f"    Used memory: {used_memory/(1024**3):.2f} GB")
    print(f"    Available memory: {available_memory/(1024**3):.2f} GB")
    print(f"    Memory used percentage: {memory.percent}%")
    if memory_used:
        print(f"    Total memory used now = {(used_memory-memory_used)/(1024**3):.2f} GB")
    print("----"*20)

    return used_memory

memory = psutil.virtual_memory()
memory_used = resources_cpu(memory)

import torch as t
from ultralytics import YOLO
from transformers import AutoFeatureExtractor, SwinModel
from torchvision import transforms

import gc
from PIL import Image

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
        print(f"    Total RAM used now = {(used_memory-memory_used)/(1024**3):.2f} GB")
    print("GPU")
    print(f"    GPU Memory = {t.cuda.memory_allocated()/(1024*1024):.2f} MB")

    print("----"*20)

    return used_memory

print("AFTER IMPORTING LIBS")
memory = psutil.virtual_memory()
memory_used = resources_gpu(memory, memory_used)

DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
model_name = "D:\\Image quality enhancement\\u_net\\models\\swin_model\\"

cls_model = YOLO("D:\\Image quality enhancement\\u_net\\models\\mo.pt").to(DEVICE)
temp_model = YOLO("D:\\Image quality enhancement\\u_net\\models\\template.pt").to(DEVICE)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
feat_model = SwinModel.from_pretrained(model_name).to(DEVICE)
ie_model = t.load("D:\\Image quality enhancement\\u_net\\models\\Image_enhancement-em.pt").to(DEVICE)
print("AFTER LOADING THE MODEL")
memory = psutil.virtual_memory()
memory_used = resources_gpu(memory, memory_used)

transforms_pipe = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ])
# img = Image.open('18526922.jpeg').convert("RGB")
img = Image.open('D:\\Image quality enhancement\\data\\X\\18518015.jpeg').convert('RGB')
img_transformed = transforms_pipe(img)
img_transformed = img_transformed.unsqueeze(0).to(DEVICE)
inputs = feature_extractor(images=img, return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
with t.inference_mode():
    res1 = cls_model('D:\\Image quality enhancement\\data\\X\\18518015.jpeg', verbose=False)
    res2 = temp_model('D:\\Image quality enhancement\\data\\X\\18518015.jpeg', verbose=False)
    res3 = feat_model(inputs['pixel_values'])
    res4 = ie_model(img_transformed)
print("AFTER PREDICTIONS")
memory = psutil.virtual_memory()
memory_used = resources_gpu(memory, memory_used)

del img, inputs, res1, res2, res3, res4, img_transformed
gc.collect()
t.cuda.empty_cache()
print("AFTER DELETING VARIABLES")
memory = psutil.virtual_memory()
memory_used = resources_gpu(memory, memory_used)

# img = Image.open('18526922.jpeg').convert("RGB")
img = Image.open('D:\\Image quality enhancement\\data\\X\\18518015.jpeg').convert('RGB')
img_transformed = transforms_pipe(img)
img_transformed = img_transformed.unsqueeze(0).to(DEVICE)
inputs = feature_extractor(images=img, return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
with t.inference_mode():
    res1 = cls_model('D:\\Image quality enhancement\\data\\X\\18518015.jpeg', verbose=False)
    res2 = temp_model('D:\\Image quality enhancement\\data\\X\\18518015.jpeg', verbose=False)
    res3 = feat_model(inputs['pixel_values'])
    res4 = ie_model(img_transformed)
print("AFTER SECOND PREDICTIONS")
memory = psutil.virtual_memory()
memory_used = resources_gpu(memory, memory_used)