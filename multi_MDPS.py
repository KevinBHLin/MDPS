import os
import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from src.models.unet import UNetModel
from src.models.resnet import Resnet
from src.dataset import MVTec, BTAD
from src.metrics import metric
from src.compare import distance
from src.diffusion import sample, sample_mask, compute_alpha

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def mdps(args):
    config = OmegaConf.load(args.config)
    print(config.data.category)
    
    unet = UNetModel(config.data.image_size, 64, dropout=0.0, n_heads=4 ,in_channels=config.data.imput_channel)
    checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category, str(config.model.ckpt)))
    unet = torch.nn.DataParallel(unet)
    unet.load_state_dict(checkpoint)    
    unet.to(config.model.device)
    unet.eval()

    if config.data.name == 'MVTec':
        test_dataset = MVTec(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=False,
    )
    if config.data.name == 'BTAD':
        test_dataset = BTAD(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=False,
        )
        
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size= config.data.batch_size,
        shuffle=False,
        num_workers= config.model.num_workers,
        drop_last=False,
    )
    
    anomaly_map_list = []
    multipul_anomaly_list = []

    resnet = Resnet(config).to(config.model.device)
    resnet.eval()
    
    if config.model.mask_steps == 0:
        labels_list = []
        predictions= []
        anomaly_map_list = []
        gt_list = []
    
        with torch.no_grad():
            for data, targets, labels in testloader:
                anomaly_batch = []
                data = data.to(config.model.device)
                test_steps = torch.Tensor([config.model.test_steps]).type(torch.int64).to(config.model.device)
                at = compute_alpha( test_steps.long(),config)
                noisy_image = at.sqrt() * data + (1- at).sqrt() * torch.randn_like(data).to('cuda')
                seq = range(0 , config.model.test_steps, config.model.skip)
                for i in range(0,config.model.test_repeat):
                    noisy_image = at.sqrt() * data + (1- at).sqrt() * torch.randn_like(data).to('cuda')
                    reconstructed = sample(data, noisy_image, seq, unet, config, w=config.model.w)
                    data_reconstructed = reconstructed[-1]
                    anomaly_map = distance(data_reconstructed, data, resnet, config)/2
                    anomaly_batch.append(anomaly_map.unsqueeze(0))
                anomaly_batch = torch.cat(anomaly_batch, dim=0)
                anomaly_map = torch.mean(anomaly_batch, dim=0) 

                transform = transforms.Compose([
                    transforms.CenterCrop((224)), 
                ]) 
                
                if config.data.name == 'MVTec':
                    anomaly_map = transform(anomaly_map)
                    targets = transform(targets)

                anomaly_map_list.append(anomaly_map)
                gt_list.append(targets)
                for pred, label in zip(anomaly_map, labels):
                    labels_list.append(0 if label == 'good' else 1)
                    k = 500
                    pred = pred.reshape(1,-1)
                    pred = F.softmax(pred, dim=1)
                    k_max, idx = pred.topk(k, largest=True)
                    score = torch.sum(k_max)
                    predictions.append(score.item())
    else:
        with torch.no_grad():
            for data, targets, labels in testloader:
                anomaly_batch = []
                data = data.to(config.model.device)
                mask_steps = torch.Tensor([config.model.mask_steps]).type(torch.int64).to(config.model.device)
                at = compute_alpha(mask_steps.long(),config)
                seq = range(0 , config.model.mask_steps, config.model.skip_mask)

                for i in range(0,config.model.mask_repeat):
                    noisy_image = at.sqrt() * data + (1- at).sqrt() * torch.randn_like(data).to('cuda')
                    reconstructed = sample(data, noisy_image, seq, unet, config, w=config.model.w_mask)
                    data_reconstructed = reconstructed[-1]
                    anomaly_map = distance(data_reconstructed, data, resnet, config)/2
                    anomaly_batch.append(anomaly_map.unsqueeze(0))
                anomaly_batch = torch.cat(anomaly_batch, dim=0)
                anomaly_map = torch.mean(anomaly_batch, dim=0) 
                anomaly_map_list.append(anomaly_map)

        anomaly_map_list = torch.cat(anomaly_map_list, dim=0)
        print(anomaly_map_list.shape)

        pixel_min = torch.min(anomaly_map_list)
        pixel_max = torch.max(anomaly_map_list)
        thresholds = pixel_min + config.model.mask0_thresholds*(pixel_max-pixel_min)
        mask_init = torch.where(anomaly_map_list > thresholds, 1, anomaly_map_list)
        mask_init = torch.where(anomaly_map_list < thresholds, 0, mask_init)

        labels_list = []
        predictions= []
        anomaly_map_list = []
        gt_list = []

        with torch.no_grad():
            for batch_idx,(data, targets, labels) in enumerate(testloader):
                anomaly_batch = []
                data = data.to(config.model.device)
                mask = mask_init[batch_idx*config.data.batch_size : batch_idx*config.data.batch_size + data.shape[0]]
                seq = range(0 , config.model.test_steps, config.model.skip)
                for i in range(0,config.model.mask_repeat):
                    reconstructed = sample_mask(data, mask, seq, unet, config, w=config.model.w)
                    data_reconstructed = reconstructed[-1]
                    anomaly_map = distance(data_reconstructed, data, resnet, config)/2
                    anomaly_batch.append(anomaly_map.unsqueeze(0))
                anomaly_batch = torch.cat(anomaly_batch, dim=0)
                multipul_anomaly_list.append(anomaly_batch)
                anomaly_map = torch.mean(anomaly_batch, dim=0)             
                transform = transforms.Compose([
                    transforms.CenterCrop((224)), 
                ])

                if config.data.name == 'MVTec':
                    anomaly_map = transform(anomaly_map)
                    targets = transform(targets)
                
                anomaly_map_list.append(anomaly_map)
                gt_list.append(targets)
                for pred, label in zip(anomaly_map, labels):
                    labels_list.append(0 if label == 'good' else 1)
                    k = 500 
                    pred = pred.reshape(1,-1)
                    pred = F.softmax(pred, dim=1)
                    k_max, idx = pred.topk(k, largest=True)
                    score = torch.sum(k_max)
                    predictions.append(score.item())
                    
    threshold,_,_ = metric(labels_list, predictions, anomaly_map_list, gt_list)

    
def parse_args():
    parser = argparse.ArgumentParser('MDPS')    
    parser.add_argument('-cfg', '--config', help='config file')
    args, unknowns = parser.parse_known_args()
    return args
    
if __name__ == "__main__":
    seed = 42
    torch.cuda.empty_cache()
    args = parse_args()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    print('*************')
    mdps(args)

