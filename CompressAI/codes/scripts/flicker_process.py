import os
import json
import imagesize
from shutil import copyfile

# directory
data_root_dir = '../../../data/flicker/flicker_2W_images'
train_root_dir = '../../../data/flicker/train'
val_root_dir = '../../../data/flicker/val'
if not os.path.exists(train_root_dir):
    os.makedirs(train_root_dir)
if not os.path.exists(val_root_dir):
    os.makedirs(val_root_dir)

# get filtered index dict
index_list = []
count = 0
for img in sorted(os.listdir(data_root_dir)):
    img_path = os.path.join(data_root_dir, img)
    width, height = imagesize.get(img_path)
    if width < 256 or height < 256:
        pass
    else:
        index_list.append(count)
    count += 1
    if count % 1000 == 0:
        print(count, 'done!')

imgs = sorted(os.listdir(data_root_dir))
alls = [imgs[i] for i in index_list]
all_length = len(alls)

train_ratio = 0.99
train_length = round(len(alls) * train_ratio)
val_length = all_length - train_length

for img_name in alls[:train_length]:
    copyfile(os.path.join(data_root_dir, img_name), os.path.join(train_root_dir, img_name))

for img_name in alls[-val_length:]:
    copyfile(os.path.join(data_root_dir, img_name), os.path.join(val_root_dir, img_name))
