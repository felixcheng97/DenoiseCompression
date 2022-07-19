from PIL import Image
import glob, os
import json
import numpy as np
import image_slicer

row_col_h_w = {
    'S6': (6, 9, 500, 592),
    'GP': (4, 8, 761, 506),
    'N6': (6, 8, 520, 526),
    'G4': (6, 8, 498, 664),
    'IP': (6, 8, 504, 504),
}

checked = {
    'S6': False,
    'GP': False,
    'N6': False,
    'G4': False,
    'IP': False,
}

annotation_dir = '../annotations'
if not os.path.exists(annotation_dir):
    os.makedirs(annotation_dir)

# prepare for train dataset
data_root = '../../../data/SIDD/SIDD_Medium_Srgb/Data'
train_dict = {}

folders = sorted(os.listdir(data_root))
for folder in folders:
    scene_instance, scene, camera, ISO, shutter_speed, temperature, brightness = folder.split('_')
    image_paths = sorted(glob.glob(os.path.join(data_root, folder, '*.PNG')))
    save_dir = os.path.join(data_root, folder).replace('SIDD_Medium_Srgb/Data', 'SIDD_Medium_Srgb_Tiles')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for image_path in image_paths:
        row, col, h, w = row_col_h_w[camera]
        # checking
        if not checked[camera]:
            img = Image.open(image_path)
            C, R = img.size
            assert R == row * h
            assert C == col * w
            checked[camera] = True
            print('camera {:s} pass checking.'.format(camera))

        print('camera {:s} checked = {}, with row = {:d}, col = {:d}'.format(camera, checked[camera], row, col))
        # tiles = image_slicer.slice(image_path, col=col, row=row, save=False)
        prefix = image_path.split('/')[-1].split('.')[0]
        print('saving {:s} tiles to {:s}\n'.format(prefix, save_dir))
        # image_slicer.save_tiles(tiles, directory=save_dir, prefix=prefix, format='png')

    train_dict[scene_instance] = {}
    gt_prefixs = sorted([image_path.split('/')[-1].split('.')[0] for image_path in image_paths if 'GT' in image_path])
    noisy_prefixs = sorted([image_path.split('/')[-1].split('.')[0] for image_path in image_paths if 'NOISY' in image_path])
    for gt_prefix, noisy_prefix in zip(gt_prefixs, noisy_prefixs):
        entry = gt_prefix.split('_')[-1]
        row, col, h, w = row_col_h_w[camera]
        train_dict[scene_instance][entry] = {
            'img_dir': save_dir,
            'gt_prefix': gt_prefix,
            'noisy_prefix': noisy_prefix,
            'row': row,
            'col': col,
            'h': h,
            'w': w,
            'H': row * h,
            'W': col * w,
        }

train_path = os.path.join(annotation_dir, 'sidd_medium_srgb_train.json')
with open(train_path, 'w') as f:
    json.dump(train_dict, f, sort_keys=True, indent=4)


# prepare for validation dataset
data_root = '../../../data/SIDD/SIDD_Benchmark_Data_Blocks/SIDD_Validation'
val_dict = {}

folders = sorted(os.listdir(data_root))
for folder in folders:
    scene_instance, scene, camera, ISO, shutter_speed, temperature, brightness = folder.split('_')
    val_dict[scene_instance] = {}
    
    image_paths = sorted(glob.glob(os.path.join(data_root, folder, '*.png')))
    gt_paths = sorted([image_path for image_path in image_paths if 'GT' in image_path])
    noisy_paths = sorted([image_path for image_path in image_paths if 'NOISY' in image_path])

    for entry, (gt_path, noisy_path) in enumerate(zip(gt_paths, noisy_paths)):
        val_dict[scene_instance]['{:03d}'.format(entry)] = {
            'gt_path': gt_path,
            'noisy_path': noisy_path,
        }

val_path = os.path.join(annotation_dir, 'sidd_medium_srgb_val.json')
with open(val_path, 'w') as f:
    json.dump(val_dict, f, sort_keys=True, indent=4)

# prepare for test dataset
data_root = '../../../data/SIDD/SIDD_Benchmark_Data_Blocks/SIDD_Test'
test_dict = {}

folders = sorted(os.listdir(data_root))
for folder in folders:
    scene_instance, scene, camera, ISO, shutter_speed, temperature, brightness = folder.split('_')
    test_dict[scene_instance] = {}
    
    image_paths = sorted(glob.glob(os.path.join(data_root, folder, '*.png')))
    noisy_paths = sorted([image_path for image_path in image_paths if 'NOISY' in image_path])

    for entry, noisy_path in enumerate(noisy_paths):
        test_dict[scene_instance]['{:03d}'.format(entry)] = {
            'noisy_path': noisy_path,
        }

test_path = os.path.join(annotation_dir, 'sidd_medium_srgb_test.json')
with open(test_path, 'w') as f:
    json.dump(test_dict, f, sort_keys=True, indent=4)