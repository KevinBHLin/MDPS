import torch
from torchmetrics import AUROC
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve

def metric(labels_list, predictions, anomaly_map_list, gt_list):
    image_auroc = roc_auc_score(labels_list, predictions)
    fpr, tpr, thresholds = roc_curve(labels_list, predictions)
    youden_j = tpr - fpr
    optimal_threshold_index = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_threshold_index]
    pixel_auroc = compute_pixel_auroc(anomaly_map_list, gt_list)
    print('AUROC: ({:.1f},{:.1f})'.format(image_auroc*100,pixel_auroc*100))
        
    return optimal_threshold, image_auroc, pixel_auroc

def compute_pixel_auroc(anomaly_map_list, gt_list):
    resutls_embeddings = anomaly_map_list[0]
    for feature in anomaly_map_list[1:]:
        resutls_embeddings = torch.cat((resutls_embeddings, feature), 0)
    resutls_embeddings =  ((resutls_embeddings - resutls_embeddings.min())/ (resutls_embeddings.max() - resutls_embeddings.min())) 
    gt_embeddings = gt_list[0]
    for feature in gt_list[1:]:
        gt_embeddings = torch.cat((gt_embeddings, feature), 0)
    resutls_embeddings = resutls_embeddings.clone().detach().requires_grad_(False)
    gt_embeddings = gt_embeddings.clone().detach().requires_grad_(False)
    auroc_p = AUROC(task="binary")
    gt_embeddings = torch.flatten(gt_embeddings).type(torch.bool).cpu().detach()
    resutls_embeddings = torch.flatten(resutls_embeddings).cpu().detach()
    pixel_auroc = auroc_p(resutls_embeddings, gt_embeddings)
    return pixel_auroc