import os

import numpy as np
import requests
import torch
import torchvision.transforms as transforms
from torch import nn
from torchvision.transforms import InterpolationMode

from .model import BiSeNet


class FaceParser(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        if weight_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            weight_path = os.path.join(base_path, "79999_iter.pth")
            if not os.path.exists(weight_path):
                download_url = "https://github.com/justsong-lab/face-parsing.PyTorch/releases/download/v0.1/79999_iter.pth"
                print(f"Downloading from: {download_url}...")
                res = requests.get(download_url)
                with open(weight_path, 'wb') as f:
                    f.write(res.content)
                print(f'Weight saved at: {weight_path}')
        self.num_classes = 19
        net = BiSeNet(n_classes=self.num_classes)
        net.cuda()
        net.load_state_dict(torch.load(weight_path))
        net.eval()
        self.net = net

        self.transform = transforms.Compose([
            transforms.Resize(512, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def get_transform(self):
        return self.transform

    @torch.no_grad()
    def forward(self, x):
        out = self.net(x)[0]
        parsing = out.argmax(1, keepdim=True)
        # parsing = np.expand_dims(parsing, 1)
        return parsing
