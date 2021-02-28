import torch

from fastai.callback.hook import hook_outputs
from torchvision import models

import torch.nn as nn
import torch.nn.functional as F

# Source: https://github.com/jantic/DeOldify/blob/master/deoldify/loss.py
class FeatureLoss(nn.Module):
    def __init__(self, layer_wgts=[20, 70, 10]):
        super().__init__()

        self.m_feat = models.vgg16_bn(True).features.cuda().eval()
        for param in self.m_feat.parameters():
            param.requires_grad = False
        blocks = [
            i - 1
            for i, o in enumerate(list(self.m_feat.children()))
            if isinstance(o, nn.MaxPool2d)
        ]
        layer_ids = blocks[2:5]
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel'] + [f'feat_{i}' for i in range(len(layer_ids))]
        self.base_loss = F.l1_loss

    def _make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self._make_features(target, clone=True)
        in_feat = self._make_features(input)
        self.feat_losses = [self.base_loss(input, target)]
        self.feat_losses += [
            self.base_loss(f_in, f_out) * w
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]

        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self):
        self.hooks.remove()
