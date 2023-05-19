import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        # TODO

        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # x: [B,3,H,W]
        # TODO
        x = self.layers(x.float())

        return x


class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        # TODO

        self.layer0 = nn.Sequential(
            nn.Conv2d(G, 8, kernel_size = 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size = 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.ConvTranspose2d(32, 16, kernel_size = 3, stride=2, padding=1, output_padding=1)
        self.layer4 = nn.ConvTranspose2d(16, 8, kernel_size = 3, stride=2, padding=1, output_padding=1)
        self.layer5 = nn.Conv2d(8, 1,kernel_size = 3, stride=1, padding=1)

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO
        B, G, D, H, W = x.size()

        x = x.transpose(1,2).reshape(B*D, G, H, W)

        C0 = self.layer0(x.float())
        C1 = self.layer1(C0)
        C2 = self.layer2(C1)
        C3 = self.layer3(C2)
        C4 = self.layer4(C3 + C1)
        out = self.layer5(C4 + C0).squeeze(1)

        return out.view(B, D, H, W)


def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B, C, H, W = src_fea.size()
    D = depth_values.size(1)

    # compute the warped positions with depth values
    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        # TODO
        points = torch.stack((x, y, torch.ones(x.shape[0]))).unsqueeze(0).repeat(B, 1, 1).double()

        depth_values = depth_values.unsqueeze(2).unsqueeze(1).repeat(1, 1, 1, H*W)
        depth_values = depth_values.float()
        rot = rot.float()
        points = points.float()
        trans = trans.float()
        p_source= torch.matmul(rot, points).unsqueeze(2).repeat(1, 1, D, 1)*depth_values + trans.unsqueeze(2)
        p_source = p_source[:, :2, :, :]/p_source[:, 2:3, :, :]
        x_norm = p_source[:, 0, :, :]/(0.5*(W - 1)) - 1
        y_norm = p_source[:, 1, :, :]/(0.5*(H - 1)) - 1
        grid = torch.stack((x_norm, y_norm), dim=3)

    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    # TODO
    warped_src_fea = F.grid_sample(src_fea.float(), grid, align_corners=True)

    return warped_src_fea.view(B, C, D, H, W)


def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    # TODO
    B, C, D, H, W = warped_src_fea.size()
    ref_fea_g = ref_fea.view(B, G, math.floor(C/G), 1, H, W)
    warped_src_fea_g = warped_src_fea.view(B, G, math.floor(C/G), D, H, W)
    similarity = (warped_src_fea_g*ref_fea_g).mean(2)

    return similarity


def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO
    B, D = depth_values.size()
    depth_values = depth_values.view(B,D,1,1)
    depth = torch.sum(p * depth_values, dim=1)

    return depth


def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO
    loss = 0
    for i in range(len(depth_est)):

        loss += F.l1_loss(depth_est[i] * mask[i], depth_gt[i] * mask[i])

    return loss
