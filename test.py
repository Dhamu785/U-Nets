import os

img_path = r'C:\Users\dhamu\Documents\Python all\alfa-TKG\AI Team\CBIR\Improvement works\Image enhancement\version-2\splits\DD\combained\data\low quality\copy\data'
files = os.listdir(img_path)
count = 0
for i in files:
    os.rename(os.path.join(img_path, i), os.path.join(img_path, str(count)+'.jpeg'))
    count += 1