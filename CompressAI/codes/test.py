import os, glob
import math
import logging
import time
import argparse
from collections import OrderedDict
import json

import torch
import torch.nn.functional as F

import numpy as np
from criterions.criterion import Criterion
import options.options as option
import utils.util as util

import compressai
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

compressai.set_entropy_coder(compressai.available_entropy_coders()[0])

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to options YMAL file.', default='./conf/test/sample.yml')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdir(opt['path']['results_root'])
util.mkdir(opt['path']['checkpoint_updated'])
util.setup_logger('base', opt['path']['log'], opt['name'], level=logging.INFO, screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### loading test model if exists
update = opt['path'].get('update', False)
if update and opt['path'].get('checkpoint', None):
    device_id = torch.cuda.current_device()
    checkpoint = torch.load(opt['path']['checkpoint'], map_location=lambda storage, loc: storage.cuda(device_id))
    model = util.create_model(opt, checkpoint, None, rank=0)
    logger.info('model checkpoint loaded from {:s}'.format(opt['path']['checkpoint']))
    model.update(force=True)
    # save the updated checkpoint
    state_dict = model.state_dict()
    for f in os.listdir(opt['path']['checkpoint_updated']):
        os.remove(os.path.join(opt['path']['checkpoint_updated'], f))
    filepath = os.path.join(opt['path']['checkpoint_updated'], opt['path']['checkpoint'].split('/')[-1])
    torch.save(state_dict, filepath)
    logger.info('updated model checkpoint saved to {:s}'.format(filepath))
else:
    try: 
        state_dict_path = os.path.join(opt['path']['checkpoint_updated'], os.listdir(opt['path']['checkpoint_updated'])[0])
        state_dict = torch.load(state_dict_path)
        model = util.create_model(opt, None, state_dict, rank=0)
        logger.info('updated model checkpoint loaded from {:s}'.format(state_dict_path))
    except:
        raise Exception('Choose not to update from a model checkpoint but fail to load from a updated model checkpoint (state_dict).')

checkpoint = None
state_dict = None
model.eval()
logger.info('Model parameter numbers: {:d}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

#### Create test dataset and dataloader
runs = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    if phase == 'train' or phase == 'val':
        pass
    else:
        device = 'cuda' if dataset_opt['cuda'] else 'cpu'
        estimation = dataset_opt['estimation']
        test_set = util.create_dataset(dataset_opt)
        test_loader = util.create_dataloader(test_set, dataset_opt, opt, None)
        logger.info('Number of test samples in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        runs.append((device, estimation, test_loader))


for device, estimation, test_loader in runs:    
    model = model.to(device)
    phase = test_loader.dataset.phase
    mode = 'est' if estimation else 'coder'
    logger.info('\nTesting [{:s}: {:s}]...'.format(mode, phase))
    save_dir = os.path.join(opt['path']['results_root'], mode, phase)
    util.mkdir(save_dir)
    for f in glob.glob(os.path.join(save_dir, '*')):
        os.remove(f)
    
    test_metrics = {
        'psnr': [],
        'ms-ssim': [],
        'bpp': [],
        'encoding_time': [],
        'decoding_time': [],
    }

    test_start_time = time.time()
    for i, data in enumerate(test_loader):
        logger.info('{:20s} - testing sample {:04d}'.format(phase, i))
        if len(data) == 1:
            gt = None
            noise = data.to(device)
            noise = util.cropping(noise)
        else:
            gt, noise = data
            gt = gt.to(device)
            gt = util.cropping(gt)
            noise = noise.to(device)
            noise = util.cropping(noise)
        
        # estimation mode using model.forward()
        if estimation:
            start = time.time()
            out_net = model.forward(noise, gt)
            elapsed_time = time.time() - start
            enc_time = dec_time = elapsed_time / 2

            num_pixels = noise.size(0) * noise.size(2) * noise.size(3)
            bpp = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in out_net["likelihoods"].values()
            )
            rec = out_net["x_hat"]
        # coder mode using model.compress() and model.decompress()
        else:
            start = time.time()
            out_enc = model.compress(noise)
            enc_time = time.time() - start

            start = time.time()
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            dec_time = time.time() - start

            num_pixels = noise.size(0) * noise.size(2) * noise.size(3)
            bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
            rec = out_dec["x_hat"]
        
        cur_psnr = util.psnr(gt, rec.clamp(0, 1)) if gt is not None else 0.
        cur_ssim = util.ms_ssim(gt, rec.clamp(0, 1), data_range=1.0).item() if gt is not None else 0.
        denoise = util.torch2img(rec[0])
        denoise.save(os.path.join(save_dir, '{:04d}_{:.3f}dB_{:.4f}_{:.4f}bpp.png'.format(i, cur_psnr, cur_ssim, bpp)))

        if gt is not None:
            gt = util.torch2img(gt[0])
            gt.save(os.path.join(save_dir, '{:04d}_gt.png'.format(i)))
        noise = util.torch2img(noise[0])
        noise.save(os.path.join(save_dir, '{:04d}_noise.png'.format(i)))

        logger.info('{:20s} - sample {:04d} image: bpp = {:.4f}, psnr = {:.3f}dB, ssim = {:.4f}'.format(phase, i, bpp, cur_psnr, cur_ssim))

        test_metrics['psnr'].append(cur_psnr)
        test_metrics['ms-ssim'].append(cur_ssim)
        test_metrics['bpp'].append(bpp)
        test_metrics['encoding_time'].append(enc_time)
        test_metrics['decoding_time'].append(dec_time)

    for k, v in test_metrics.items():
        test_metrics[k] = [sum(v) / len(v)]
    logger.info('----Average results for phase {:s}----'.format(phase))
    for k, v in test_metrics.items():
        logger.info('\t{:s}: {:.4f}'.format(k, v[0]))

    test_end_time = time.time()
    logger.info('Total testing time for phase {:s} = {:.3f}s'.format(phase, test_end_time - test_start_time))

    # save results
    description = "entropy estimation" if estimation else "ans"
    output = {
        "name": '{:s}_{:s}'.format(opt['name'], phase),
        "description": f"Inference ({description})",
        "results": test_metrics,
    }
    json_path = os.path.join(opt['path']['results_root'], mode, '{:s}.json'.format(phase))
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
