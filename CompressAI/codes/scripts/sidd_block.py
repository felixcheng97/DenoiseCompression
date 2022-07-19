import os
import scipy.io
from PIL import Image

val_noisy_blocks_path = '../../../data/SIDD/SIDD_Benchmark_zips/ValidationNoisyBlocksSrgb.mat'
val_noisy_blocks = scipy.io.loadmat(val_noisy_blocks_path)
val_noisy_blocks_numpy = val_noisy_blocks['ValidationNoisyBlocksSrgb']

val_gt_blocks_path = '../../../data/SIDD/SIDD_Benchmark_zips/ValidationGtBlocksSrgb.mat'
val_gt_blocks = scipy.io.loadmat(val_gt_blocks_path)
val_gt_blocks_numpy = val_gt_blocks['ValidationGtBlocksSrgb']

test_noisy_blocks_path = '../../../data/SIDD/SIDD_Benchmark_zips/BenchmarkNoisyBlocksSrgb.mat'
test_noisy_blocks = scipy.io.loadmat(test_noisy_blocks_path)
test_noisy_blocks_numpy = test_noisy_blocks['BenchmarkNoisyBlocksSrgb']

val_dir = '../../../data/SIDD/SIDD_Benchmark_Data_Blocks/SIDD_Validation'
test_dir = '../../../data/SIDD/SIDD_Benchmark_Data_Blocks/SIDD_Test'

num_of_scenes = 40
num_of_blocks = 32
scene_names = sorted(os.listdir('../../../data/SIDD/SIDD_Benchmark_Data'))
for idx in range(num_of_scenes):
    scene_name = scene_names[idx]
    scene = scene_name.split('_')[0]

    val_scene_dir = os.path.join(val_dir, scene_name)
    if not os.path.exists(val_scene_dir):
        os.makedirs(val_scene_dir)
    test_scene_dir = os.path.join(test_dir, scene_name)
    if not os.path.exists(test_scene_dir):
        os.makedirs(test_scene_dir)

    for block in range(num_of_blocks):
        val_noisy_img = Image.fromarray(val_noisy_blocks_numpy[idx, block])
        val_noisy_path = os.path.join(val_scene_dir, '{:s}_NOISY_SRGB_{:02d}.png'.format(scene, block))
        val_noisy_img.save(val_noisy_path)
        
        val_gt_img = Image.fromarray(val_gt_blocks_numpy[idx, block])
        val_gt_path = os.path.join(val_scene_dir, '{:s}_GT_SRGB_{:02d}.png'.format(scene, block))
        val_gt_img.save(val_gt_path)

        test_noisy_img = Image.fromarray(test_noisy_blocks_numpy[idx, block])
        test_noisy_path = os.path.join(test_scene_dir, '{:s}_NOISY_SRGB_{:02d}.png'.format(scene, block))
        test_noisy_img.save(test_noisy_path)