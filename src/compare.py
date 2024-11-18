import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from torchvision.transforms import transforms
import math 

def distance(input1, input2, resnet, config):
    sigma = 4
    kernel_size = 2 * int(4 * sigma + 0.5) + 1
    anomaly_map = 0
    device = config.model.device
    input1 = input1.to(device)
    input2 = input2.to(device)
    i_d = MSE(input1, input2)
    f_d = LPIPS(input1, input2, resnet, config)
    f_d = torch.Tensor(f_d).to(device)
    max_f_d = torch.max(f_d)
    max_i_d = torch.max(i_d)
    anomaly_map += f_d + config.model.eta * max_f_d/max_i_d * i_d
    
    anomaly_map = gaussian_blur2d(anomaly_map, 
                                  kernel_size=(kernel_size, kernel_size), 
                                  sigma=(sigma, sigma))
    anomaly_map = torch.sum(anomaly_map, dim=1).unsqueeze(1)
    return anomaly_map

def MSE(output, target):
    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    output = transform(output)
    target = transform(target)
    distance_map = torch.mean(torch.abs(output - target), dim=1).unsqueeze(1)
    return distance_map

def LPIPS(output, target, resnet, config):
    resnet.eval()
    def normalize(tensor):
        return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])((tensor + 1) / 2)
    
    target_normalized = normalize(target)
    output_normalized = normalize(output)
    target_features = resnet(target_normalized)
    output_features = resnet(output_normalized)
    
    out_size = config.data.image_size
    device = config.model.device
    anomaly_map = torch.zeros(target_features[0].shape[0], 1, out_size, out_size, device=device)
    for i in range(1, len(target_features)):
        target_patches = patchify(target_features[i])
        output_patches = patchify(output_features[i])
        similarity_map = F.cosine_similarity(target_patches, output_patches)
        a_map = 1 - similarity_map
        a_map = a_map.unsqueeze(dim=1)
        interpolated_a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        anomaly_map += interpolated_a_map
    
    return anomaly_map


def patchify(features, return_spatial_info=False):
    patchsize = 3
    stride = 1
    padding = int((patchsize - 1) / 2)
    unfolder = torch.nn.Unfold(
        kernel_size=patchsize, stride=stride, padding=padding, dilation=1
    )
    unfolded_features = unfolder(features)
    number_of_total_patches = []
    for s in features.shape[-2:]:
        n_patches = (
            s + 2 * padding - 1 * (patchsize - 1) - 1
        ) / stride + 1
        number_of_total_patches.append(int(n_patches))
    unfolded_features = unfolded_features.reshape(
        *features.shape[:2], patchsize, patchsize, -1
    )
    unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
    max_features = torch.mean(unfolded_features, dim=(3,4))
    features = max_features.reshape(features.shape[0], int(math.sqrt(max_features.shape[1])) , int(math.sqrt(max_features.shape[1])), max_features.shape[-1]).permute(0,3,1,2)
    if return_spatial_info:
        return unfolded_features, number_of_total_patches
    return features
