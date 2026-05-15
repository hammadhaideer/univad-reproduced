import torch
import torch.nn as nn
from utils.comad_utils import *
from models.dino import vision_transformer as vits


class DinoFeaturizer(nn.Module):
    def __init__(self, device: str = "cuda", pretrained_ckpts: str = "pretrained_ckpts"):
        super().__init__()
        patch_size      = 8
        self.patch_size = patch_size
        self.feat_type  = "feat"
        self.device     = torch.device(device)
        arch            = "vit_small"

        self.model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False

        if arch == "vit_small" and patch_size == 16:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError(f"Unknown arch={arch} and patch_size={patch_size}")

        state_dict = torch.hub.load_state_dict_from_url(
            url=url,
            model_dir=pretrained_ckpts,
            map_location="cpu",
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()

        self.n_feats = 384 if arch == "vit_small" else 768

    def forward(self, img, n=1, return_class_feat=False):
        self.model.eval()
        with torch.no_grad():
            assert img.shape[2] % self.patch_size == 0
            assert img.shape[3] % self.patch_size == 0

            feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            if self.feat_type == "feat":
                image_feat = (
                    feat[:, 1:, :]
                    .reshape(feat.shape[0], feat_h, feat_w, -1)
                    .permute(0, 3, 1, 2)
                )
            elif self.feat_type == "KK":
                image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
                B, H, I, J, D = image_k.shape
                image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            else:
                raise ValueError(f"Unknown feat_type: {self.feat_type}")

            if return_class_feat:
                return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)

        return image_feat, image_feat


class LinearLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = tokens[i][:, 1:, :]
            else:
                raise ValueError(f"Unexpected token shape at index {i}: {tokens[i].shape}")
        return tokens
