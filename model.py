
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

class ColorizationDataset(Dataset):
    def __init__(self, gray_dir, color_dir):
        self.gray_dir = gray_dir
        self.color_dir = color_dir
        self.gray_images = sorted(os.listdir(gray_dir))
        self.color_images = sorted(os.listdir(color_dir))

    def __len__(self):
        return len(self.gray_images)

    def __getitem__(self, idx):
        gray_path = os.path.join(self.gray_dir, self.gray_images[idx])
        color_path = os.path.join(self.color_dir, self.color_images[idx])

        gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(color_path)
        gray = cv2.resize(gray, (128, 128))
        color = cv2.resize(color, (128, 128))

        # Normalize to [0,1]
        gray = gray.astype(np.float32) / 255.0
        color = color.astype(np.float32) / 255.0

        # Convert color to LAB
        # lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
        # L = lab[:, :, 0:1] / 255.0
        # ab = lab[:, :, 1:3] / 255.0
        lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0:1] / 100.0           # normalize L to [0,1]
        ab = lab[:, :, 1:3] / 128.0          # normalize ab to [-1,1]


        L = torch.from_numpy(L.transpose((2, 0, 1))).float()
        ab = torch.from_numpy(ab.transpose((2, 0, 1))).float()

        return L, ab

class UNetColorization(nn.Module):
    def __init__(self):
        super(UNetColorization, self).__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.enc1 = nn.Sequential(CBR(1, 64), CBR(64, 64))
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.enc4 = nn.Sequential(CBR(256, 512), CBR(512, 512))

        self.bottleneck = nn.Sequential(CBR(512, 1024), CBR(1024, 512))

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec3 = nn.Sequential(CBR(512 + 512, 512), CBR(512, 256))
        self.dec2 = nn.Sequential(CBR(256 + 256, 256), CBR(256, 128))
        self.dec1 = nn.Sequential(CBR(128 + 128, 128), CBR(128, 64))
        self.final_conv = nn.Conv2d(64 + 64, 2, 1)


    def forward(self, x):
        e1 = self.enc1(x)                  # 256x256
        e2 = self.enc2(self.pool(e1))      # 128x128
        e3 = self.enc3(self.pool(e2))      # 64x64
        e4 = self.enc4(self.pool(e3))      # 32x32

        b = self.bottleneck(self.pool(e4)) # 16x16

        d4 = self.up(b)                    # 32x32
        d4 = self.dec3(torch.cat([d4, e4], dim=1))

        d3 = self.up(d4)                   # 64x64
        d3 = self.dec2(torch.cat([d3, e3], dim=1))

        d2 = self.up(d3)                   # 128x128
        d2 = self.dec1(torch.cat([d2, e2], dim=1))

        d1 = self.up(d2)                   # 256x256
        out = self.final_conv(torch.cat([d1, e1], dim=1))
        out = torch.tanh(out)
        return out

def lab_to_rgb(L, ab):
    # Convert from tensors to numpy
    L = L.squeeze().cpu().numpy()
    ab = ab.squeeze().cpu().numpy()

    # Rescale back to proper LAB ranges
    L = L * 100  # [0, 100]
    ab = ab * 128  # [-128, 128]

    # Combine into LAB image
    lab = np.concatenate((L[np.newaxis, :, :], ab), axis=0).transpose((1, 2, 0)).astype(np.float32)

    # Convert LAB → RGB correctly
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Clip to [0,1] for display
    rgb = np.clip(rgb, 0, 1)
    return rgb