from __future__ import absolute_import, division, print_function

import torch

import torch.nn.functional as F


def _elementwise_epe(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=2, dim=1, keepdim=True)

def _elementwise_robust_epe_char(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.pow(torch.norm(residual, p=2, dim=1, keepdim=True) + 0.01, 0.4)

def _downsample2d_as(inputs, target_as):

    _, _, h, w = target_as.size()

    return F.adaptive_avg_pool2d(inputs, [h, w])

def _upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return F.interpolate(inputs, [h, w], mode=mode, align_corners=True)

def f1_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, 1)

def fbeta_score(y_true, y_pred, beta, eps=1e-8):
    beta2 = beta ** 2

    y_pred = y_pred.float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=2).sum(dim=2)
    precision = true_positive / (y_pred.sum(dim=2).sum(dim=2) + eps)
    recall = true_positive / (y_true.sum(dim=2).sum(dim=2) + eps)

    return torch.mean(precision * recall / (precision * beta2 + recall + eps) * (1 + beta2))

def f1_score_bal_loss(y_pred, y_true):
    eps = 1e-8

    tp = -(y_true * torch.log(y_pred + eps)).sum(dim=2).sum(dim=2).sum(dim=1)
    fn = -((1 - y_true) * torch.log((1 - y_pred) + eps)).sum(dim=2).sum(dim=2).sum(dim=1)

    denom_tp = y_true.sum(dim=2).sum(dim=2).sum(dim=1) + y_pred.sum(dim=2).sum(dim=2).sum(dim=1) + eps
    denom_fn = (1 - y_true).sum(dim=2).sum(dim=2).sum(dim=1) + (1 - y_pred).sum(dim=2).sum(dim=2).sum(dim=1) + eps

    return ((tp / denom_tp).sum() + (fn / denom_fn).sum()) * y_pred.size(2) * y_pred.size(3) * 0.5
def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()


def realEPE(output, target):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
    return EPE(upsampled_output, target)



def sequence_loss(flow_preds, flow_gt,gamma=0.8):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        # i_loss = torch.sum((flow_preds[i] - flow_gt) ** 2, dim=1).sqrt().mean()
        i_loss = (flow_preds[i] -  flow_gt).abs()
        # i_loss=_elementwise_epe(flow_preds[i] ,flow_gt).sum()
        # i_loss = F.smooth_l1_loss(flow_preds[i] ,flow_gt, size_average=True)

        flow_loss += i_weight * i_loss.mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()

    metrics = epe.mean()



    return flow_loss, metrics










