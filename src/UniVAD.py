import os
import warnings
from enum import Enum
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from torch import nn
from torchvision.transforms import v2

from models.clip_prompt import encode_text_with_prompt_ensemble
from models.component_feature_extractor import ComponentFeatureExtractor
from models.component_segmentaion import (
    split_masks_from_one_mask,
    split_masks_from_one_mask_torch,
    split_masks_from_one_mask_with_bg,
)
from modules import DinoFeaturizer, LinearLayer
from utils.crf import dense_crf
from utils.filter_algorithm import filter_bg_noise
from utils.sampler import GreedyCoresetSampler
import models.clip as open_clip


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_i_m   = np.array(IMAGENET_MEAN)[:, None, None]
_i_std = np.array(IMAGENET_STD)[:, None, None]

# Maximum K-means retries before giving up and using the best attempt found.
_KMEANS_MAX_RETRIES = 20


class object_type(Enum):
    TEXTURE = 0
    SINGLE  = 1
    MULTI   = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rel_path_from_data(image_path: str) -> str:
    """
    Extract the dataset-relative portion of an image path in a
    cross-platform way.

    Given /some/prefix/data/mvtec/bottle/test/broken/000.png
    returns  mvtec/bottle/test/broken/000.png

    Falls back to the filename alone if '/data/' is not found.
    """
    # Normalise separators so the split works on Windows too.
    normalised = image_path.replace("\\", "/")
    parts = normalised.split("/data/")
    return parts[-1] if len(parts) > 1 else Path(image_path).name


def get_heatmaps(img, query_feature, net, color_tensor, device="cuda", use_crf=True):
    """Compute component heatmaps using DINO features.

    Parameters
    ----------
    img:           Input image tensor on the correct device.
    query_feature: Normal reference feature tensor.
    net:           DinoFeaturizer network.
    color_tensor:  Colour palette for visualisation.
    device:        Torch device string — must match img.device.
    use_crf:       Whether to apply dense CRF post-processing.
                   Set False on platforms without pydensecrf.
    """
    with torch.no_grad():
        feats1, _ = net(img.to(device))

    attn_intra = torch.einsum(
        "nchw,ncij->nhwij",
        F.normalize(query_feature, dim=1),
        F.normalize(feats1, dim=1),
    )
    attn_intra -= attn_intra.mean([3, 4], keepdims=True)
    attn_intra  = attn_intra.clamp(0).squeeze(0)

    heatmap_intra = (
        F.interpolate(
            attn_intra, img.shape[2:], mode="bilinear", align_corners=True
        )
        .squeeze(0)
        .detach()
        .cpu()
    )

    if use_crf:
        img_crf      = img.squeeze()
        crf_result   = dense_crf(img_crf, heatmap_intra)
        heatmap_intra = torch.from_numpy(crf_result)

    heatmap = heatmap_intra.argmax(dim=0)
    return heatmap, heatmap_intra


def _save_debug_image(data, savepath, heatmap_intra):
    """Write the input image and per-component heatmaps to disk."""
    img_np = data[0].cpu().numpy()
    img_np = np.clip((img_np * _i_std + _i_m) * 255, 0, 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np.transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(savepath, "img.jpg"), img_np)

    for i in range(heatmap_intra.shape[0]):
        heat = np.round(
            heatmap_intra[i].cpu().numpy() * 128
        ).astype(np.uint8)
        cv2.imwrite(os.path.join(savepath, f"heatresult{i}.jpg"), heat)


# ---------------------------------------------------------------------------
# UniVAD
# ---------------------------------------------------------------------------

class UniVAD(nn.Module):
    """
    UniVAD: Training-free unified few-shot visual anomaly detection.

    Gu et al., CVPR 2025 — arXiv:2412.03342

    Parameters
    ----------
    image_size:       Input resolution (default 448, as used in the paper).
    device:           Torch device string, e.g. "cuda" or "cpu".
    backbone:         DINOv2 variant to load from models/dinov2.
                      "dinov2_vitg14" matches the paper.
                      "dinov2_vitb14" for runs on 16 GB GPUs.
    use_crf:          Apply dense CRF post-processing in heatmap computation.
                      Requires pydensecrf. Set False on Windows / Mac.
    masks_root:       Directory containing pre-computed Grounded SAM masks.
    heat_masks_root:  Directory for runtime component heatmask outputs.
    """

    def __init__(
        self,
        image_size:      int  = 448,
        device:          str  = "cuda",
        backbone:        str  = "dinov2_vitg14",
        use_crf:         bool = True,
        masks_root:      str  = "masks",
        heat_masks_root:  str  = "heat_masks",
        pretrained_ckpts: str  = "pretrained_ckpts",
    ) -> None:
        super().__init__()

        self.image_size      = image_size
        self.device          = torch.device(device)
        self.use_crf         = use_crf
        self.masks_root      = masks_root
        self.heat_masks_root  = heat_masks_root
        self.pretrained_ckpts = pretrained_ckpts

        # ── CLIP ────────────────────────────────────────────────────────
        clip_name = "ViT-L-14-336"
        self.out_layers = [6, 12, 18, 24]

        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_name, image_size, pretrained="openai"
        )
        self.clip_model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(clip_name)

        # ── DINOv2 ──────────────────────────────────────────────────────
        self.dino_net = DinoFeaturizer(device=str(self.device), pretrained_ckpts=self.pretrained_ckpts).eval()
        self.dinov2_net = torch.hub.load(
            "./models/dinov2", backbone, pretrained=True, source="local"
        ).to(self.device).eval()

        # ── Remaining modules ────────────────────────────────────────────
        self.cfa     = CFA()
        self.decoder = LinearLayer()

        # ── Transforms ──────────────────────────────────────────────────
        self.transform_clip = v2.Compose([
            v2.Resize((image_size, image_size)),
            v2.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        self.transform_dino = v2.Compose([
            v2.Resize((image_size, image_size)),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.just_resize = v2.Resize((image_size, image_size))

        # Transforms used by ComponentFeatureExtractor (no ToTensor step —
        # the extractor receives a pre-converted numpy array).
        _ce_clip = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        _ce_dino = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        com_config = {"transform_clip": _ce_clip, "transform_dino": _ce_dino}
        self.component_feature_extractor = ComponentFeatureExtractor(
            com_config,
            clip_model=self.clip_model,
            dino_model=self.dinov2_net,
        )

        with torch.no_grad():
            self.text_prompts = encode_text_with_prompt_ensemble(
                self.clip_model, ["object"], self.tokenizer, self.device
            )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _mask_path(self, image_path: str) -> Path:
        """Build the Grounded SAM mask path for a given image."""
        rel = _rel_path_from_data(image_path)
        # Strip original extension and attach grounding_mask sub-path.
        rel_no_ext = Path(rel).with_suffix("")
        return Path(self.masks_root) / rel_no_ext / "grounding_mask.png"

    def _heat_path(self, tag: str) -> Path:
        """Return and create a heat_masks sub-directory."""
        path = Path(self.heat_masks_root) / tag
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -----------------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        batch: torch.Tensor,
        image_path: str,
        image_pil=None,
    ) -> dict[str, torch.Tensor]:

        clip_img = self.transform_clip(batch)
        dino_img = self.transform_dino(batch)

        with torch.no_grad():
            image_features, patch_tokens = self.clip_model.encode_image(
                clip_img, self.out_layers
            )
            image_features = image_features[:, 0, :]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            patch_tokens   = self.decoder(patch_tokens)
            dino_tokens    = self.dinov2_net.forward_features(dino_img)["x_norm_patchtokens"]
            text_features  = self.text_prompts["object"]

        # Infer dimensions from actual tensors — never hardcoded.
        dino_dim = dino_tokens.shape[-1]
        clip_dim = patch_tokens[self.out_layers[0]].shape[-1] if patch_tokens else 1024

        # ── Global image-level score (CLIP) ─────────────────────────────
        global_score = (
            1 - (image_features @ self.normal_image_features.transpose(-2, -1)).max().item()
        )

        # ── CLIP patch-level anomaly map (CAPM) ─────────────────────────
        patch_grid = int((self.image_size / 14) ** 2)
        sims = []
        for i in range(len(patch_tokens)):
            if i % 2 == 0:
                continue
            q = patch_tokens[i].view(patch_grid, 1, clip_dim)
            k = self.normal_patch_tokens[i].reshape(1, -1, clip_dim)
            sim_max = F.cosine_similarity(q, k, dim=2).max(dim=1).values
            sims.append(sim_max)
        sim_clip = torch.mean(torch.stack(sims, dim=0), dim=0)
        sim_clip = sim_clip.reshape(1, 1, self.image_size // 14, self.image_size // 14)
        sim_clip = F.interpolate(sim_clip, size=self.image_size, mode="bilinear", align_corners=True)
        anomaly_map_clip = 1 - sim_clip

        # ── DINOv2 patch-level anomaly map ──────────────────────────────
        dino_q = dino_tokens.view(-1, 1, dino_dim)
        dino_k = self.normal_dino_patches.reshape(1, -1, dino_dim)
        sim_dino = F.cosine_similarity(dino_q, dino_k, dim=2).max(dim=1).values
        sim_dino = sim_dino.reshape(1, 1, self.image_size // 14, self.image_size // 14)
        sim_dino = F.interpolate(sim_dino, size=self.image_size, mode="bilinear", align_corners=True)
        anomaly_map_dino = 1 - sim_dino

        # ── Vision-language anomaly map (GECM text branch) ──────────────
        vl_maps = []
        for layer in range(len(patch_tokens)):
            if layer != 6:
                continue
            pt = patch_tokens[layer] @ self.clip_model.visual.proj
            pt = pt / pt.norm(dim=-1, keepdim=True)
            vl  = 100.0 * pt @ text_features
            B, L, C = vl.shape
            H_grid = int(np.sqrt(L))
            vl = F.interpolate(
                vl.permute(0, 2, 1).view(B, 2, H_grid, H_grid),
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            )
            vl = torch.softmax(vl, dim=1)
            vl = (vl[:, 1] - vl[:, 0] + 1) / 2
            vl_maps.append(vl)
        anomaly_map_vl = torch.mean(torch.stack(vl_maps, dim=0), dim=0).unsqueeze(1)

        # ── TEXTURE gate — return early ──────────────────────────────────
        if self.gate == object_type.TEXTURE:
            anomaly_map = (anomaly_map_clip + anomaly_map_dino + anomaly_map_vl) / 3
            score_fn = "mean" if "HIS" in image_path else "max"
            pred_score = anomaly_map.mean() if score_fn == "mean" else anomaly_map.max()
            return {
                "pred_score": torch.tensor(pred_score.item()),
                "pred_mask":  anomaly_map,
            }

        # ── Load query SAM masks for SINGLE / MULTI ──────────────────────
        mask_path = self._mask_path(image_path)
        query_tmp = np.array(
            Image.open(mask_path).resize((self.image_size, self.image_size))
        )
        query_sam_masks = split_masks_from_one_mask_torch(torch.tensor(query_tmp))
        if not query_sam_masks:
            query_sam_masks = [torch.ones((self.image_size, self.image_size))]

        # ── SINGLE gate ──────────────────────────────────────────────────
        if self.gate == object_type.SINGLE:
            anomaly_part_clip = torch.zeros(
                1, 1, self.image_size // 14, self.image_size // 14,
                device=self.device,
            )
            anomaly_part_dino = torch.zeros_like(anomaly_part_clip)

            for sam_mask in query_sam_masks:
                h, w    = sam_mask.shape
                kernel  = np.ones((5, 5), np.uint8)
                dilated = cv2.dilate(np.array(sam_mask), kernel, iterations=1)
                thresh  = torch.tensor(dilated).reshape(1, 1, h, w)
                thresh  = F.interpolate(
                    thresh, size=self.image_size // 14,
                    mode="bilinear", align_corners=True,
                ).reshape(-1)
                thresh[thresh > 0] = 1
                mask_idx = thresh > 0

                # CLIP part
                sims = []
                for i in range(len(patch_tokens)):
                    if i % 2 == 0:
                        continue
                    q = patch_tokens[i].view(patch_grid, 1, clip_dim)[mask_idx]
                    k = self.normal_clip_part_patch_features[i][0].reshape(1, -1, clip_dim)
                    sims.append(F.cosine_similarity(q, k, dim=2).max(dim=1).values)
                sim_part_clip = torch.mean(torch.stack(sims, dim=0), dim=0)

                # DINOv2 part
                q_dino = dino_tokens.view(-1, 1, dino_dim)[mask_idx]
                k_dino = self.normal_dino_part_patch_features[0].reshape(1, -1, dino_dim)
                sim_part_dino = F.cosine_similarity(q_dino, k_dino, dim=2).max(dim=1).values

                thresh_2d = thresh.reshape(1, 1, self.image_size // 14, self.image_size // 14)
                anomaly_part_clip[thresh_2d > 0] += 1 - sim_part_clip
                anomaly_part_dino[thresh_2d > 0] += 1 - sim_part_dino

            anomaly_part_clip = F.interpolate(
                anomaly_part_clip, size=self.image_size, mode="bilinear", align_corners=True
            )
            anomaly_part_dino = F.interpolate(
                anomaly_part_dino, size=self.image_size, mode="bilinear", align_corners=True
            )
            anomaly_map = (
                (anomaly_map_clip + anomaly_map_dino) / 2
                + (anomaly_part_clip + anomaly_part_dino) / 2
                + anomaly_map_vl
            ) / 3
            return {
                "pred_score": torch.tensor(anomaly_map.max().item() + global_score),
                "pred_mask":  anomaly_map,
            }

        # ── MULTI gate ───────────────────────────────────────────────────
        heatmap, heatmap_intra = get_heatmaps(
            dino_img,
            self.train_features_sampled,
            self.dino_net,
            None,
            device=str(self.device),
            use_crf=self.use_crf,
        )

        gs_masks      = torch.stack(query_sam_masks)
        cluster_masks = torch.stack(split_masks_from_one_mask_torch(heatmap + 1))
        heatmap_ref   = assign_fine_to_coarse_torch(cluster_masks, gs_masks).max(dim=0).values

        savepath = self._heat_path(f"{self.class_name}_heat/test/0")
        cv2.imwrite(str(savepath / "heatresult_refined.png"), heatmap_ref.numpy())

        query_tmp_mask     = cv2.imread(str(savepath / "heatresult_refined.png"), cv2.IMREAD_GRAYSCALE)
        query_masks_capm, query_mask_idxs = split_masks_from_one_mask_with_bg(query_tmp_mask)
        query_masks, _     = split_masks_from_one_mask(query_tmp_mask)

        kernel = np.ones((5, 5), np.uint8)
        query_masks_capm   = [cv2.dilate(m, kernel, iterations=1) for m in query_masks_capm]
        query_masks        = [cv2.dilate(m, kernel, iterations=1) for m in query_masks]

        _sentinel = 100
        anomaly_part_clip = torch.full(
            (1, 1, self.image_size // 14, self.image_size // 14),
            _sentinel, device=self.device
        )
        anomaly_part_dino = torch.full_like(anomaly_part_clip, _sentinel)

        for j in range(len(query_masks_capm)):
            h, w   = query_masks_capm[j].shape
            thresh = torch.tensor(query_masks_capm[j]).reshape(1, 1, h, w)
            thresh = F.interpolate(
                thresh, size=self.image_size // 14,
                mode="bilinear", align_corners=True,
            ).reshape(-1)
            if thresh.sum() < 1:
                continue
            thresh[thresh > 0] = 1
            mask_idx = thresh > 0

            if not self.normal_dino_part_patch_features[query_mask_idxs[j]]:
                continue

            sims = []
            for i in range(len(patch_tokens)):
                if i % 2 == 0:
                    continue
                q = patch_tokens[i].view(patch_grid, 1, clip_dim)[mask_idx]
                k = self.normal_clip_part_patch_features[i][query_mask_idxs[j]].reshape(1, -1, clip_dim)
                sims.append(F.cosine_similarity(q, k, dim=2).max(dim=1).values)
            sim_part_clip = torch.mean(torch.stack(sims, dim=0), dim=0)

            q_dino        = dino_tokens.view(-1, 1, dino_dim)[mask_idx]
            k_dino        = self.normal_dino_part_patch_features[query_mask_idxs[j]].reshape(1, -1, dino_dim)
            sim_part_dino = F.cosine_similarity(q_dino, k_dino, dim=2).max(dim=1).values

            thresh_2d = thresh.reshape(1, 1, self.image_size // 14, self.image_size // 14)
            anomaly_part_clip[thresh_2d > 0] = torch.min(
                1 - sim_part_clip, anomaly_part_clip[thresh_2d > 0]
            )
            anomaly_part_dino[thresh_2d > 0] = torch.min(
                1 - sim_part_dino, anomaly_part_dino[thresh_2d > 0]
            )

        anomaly_part_clip[anomaly_part_clip == _sentinel] = 0
        anomaly_part_dino[anomaly_part_dino == _sentinel] = 0
        anomaly_part_clip = F.interpolate(
            anomaly_part_clip, size=self.image_size, mode="bilinear", align_corners=True
        )
        anomaly_part_dino = F.interpolate(
            anomaly_part_dino, size=self.image_size, mode="bilinear", align_corners=True
        )

        # GECM component-level features
        if image_pil is not None:
            image_np = np.array(image_pil[0])
        else:
            image_np = np.array(
                Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
            )

        features = self.component_feature_extractor.extract(image_np, query_masks)
        qcf = {k: [] for k in ["area", "color", "position", "clip_image", "dino_image", "geo"]}
        for key in ["area", "color", "position", "clip_image", "dino_image"]:
            qcf[key] = torch.cat([features[key]], axis=0)
        qcf["geo"] = torch.cat([qcf["area"], qcf["color"], qcf["position"]], dim=1)
        qcf["clip_image"] = qcf["clip_image"].transpose(0, 1)

        for layer in range(qcf["clip_image"].shape[0]):
            qcf["clip_image"][layer] = self.cfa(qcf["clip_image"][layer])
        qcf["dino_image"] = self.cfa(qcf["dino_image"])

        anomaly_map_dist = torch.zeros(1, 1, self.image_size, self.image_size, device=self.device)
        for mask_idx in range(len(query_masks)):
            thresh_ori = torch.tensor(query_masks[mask_idx]).reshape(1, 1, self.image_size, self.image_size)

            sim_clip_cmp = F.cosine_similarity(
                qcf["clip_image"][:, mask_idx].unsqueeze(1).unsqueeze(1),
                self.normal_component_feats["clip_image"].unsqueeze(1),
                dim=-1,
            )
            sim_dino_cmp = F.cosine_similarity(
                qcf["dino_image"][mask_idx].unsqueeze(0),
                self.normal_component_feats["dino_image"],
                dim=1,
            )
            sim_geo = F.cosine_similarity(
                qcf["geo"][mask_idx],
                self.normal_component_feats["geo"].unsqueeze(0),
                dim=2,
            )
            dist  = torch.mean(1 - sim_clip_cmp.max(dim=-1).values, dim=0).item()
            dist += 1 - sim_dino_cmp.max().item()
            dist += 1 - sim_geo.max().item()
            anomaly_map_dist[thresh_ori > 0] += dist

        anomaly_map = (
            (anomaly_map_clip + anomaly_map_dino) / 2
            + (anomaly_part_clip + anomaly_part_dino) / 2
            + anomaly_map_vl
        ) / 3 + anomaly_map_dist / 2

        return {
            "pred_score": torch.tensor(anomaly_map.max().item() + global_score),
            "pred_mask":  anomaly_map,
        }

    # -----------------------------------------------------------------------
    # setup
    # -----------------------------------------------------------------------

    def setup(self, data: dict, re_seg: bool = True) -> None:
        """
        Initialise reference memory banks from a small set of normal images.

        Parameters
        ----------
        data:    Dict with keys:
                   few_shot_samples  — Tensor[K, C, H, W] of normal reference images.
                   dataset_category  — Category name string.
                   image_path        — List of K file paths for the reference images.
        re_seg:  If False, load cached component clustering from disk
                 instead of re-running K-means segmentation.
        """
        few_shot   = data["few_shot_samples"]
        self.class_name = data["dataset_category"]
        image_paths = data["image_path"]

        self.kernel = np.ones((20, 20), np.uint8)
        self.shot   = len(few_shot)

        clip_normal = self.transform_clip(few_shot).to(self.device)
        dino_normal = self.transform_dino(few_shot).to(self.device)

        # ── Load Grounded SAM masks for reference images ─────────────────
        grounded_sam_mask_paths = []
        for ip in image_paths:
            mask_p = self._mask_path(ip)
            grounded_sam_mask_paths.append(str(mask_p))

        grounded_sam_masks = []
        for mp in grounded_sam_mask_paths:
            raw = np.array(Image.open(mp).resize((self.image_size, self.image_size)))
            grounded_sam_masks.append(
                split_masks_from_one_mask_torch(torch.tensor(raw))
            )

        # Safe defaults for H, W — overwritten below when masks are available.
        H = W = self.image_size

        if grounded_sam_masks[0]:
            H, W       = grounded_sam_masks[0][0].shape
            largest    = sorted(grounded_sam_masks[0], key=lambda x: x.sum(), reverse=True)[0]
            obj_ratio  = (largest.sum().item() / 255) / (H * W)
        else:
            obj_ratio = 1.0

        if obj_ratio > 0.65 and len(grounded_sam_masks[0]) <= 2:
            self.gate = object_type.TEXTURE
        elif len(grounded_sam_masks[0]) == 1:
            self.gate = object_type.SINGLE
        else:
            self.gate = object_type.MULTI

        # ── Build CLIP and DINOv2 reference banks ────────────────────────
        with torch.no_grad():
            self.normal_image_features, self.normal_patch_tokens = (
                self.clip_model.encode_image(clip_normal, self.out_layers)
            )
            self.normal_image_features = self.normal_image_features[:, 0, :]
            self.normal_image_features = (
                self.normal_image_features / self.normal_image_features.norm()
            )
            self.normal_patch_tokens = self.decoder(self.normal_patch_tokens)
            self.normal_dino_patches = self.dinov2_net.forward_features(
                dino_normal
            )["x_norm_patchtokens"]

        # Infer dims from actual tensors.
        dino_dim = self.normal_dino_patches.shape[-1]

        # ── Initialise part-feature storage ─────────────────────────────
        n_parts = 10
        self.normal_dino_part_patch_features = [[] for _ in range(n_parts)]
        self.normal_clip_part_patch_features = [
            [[] for _ in range(n_parts)]
            for _ in range(len(self.normal_patch_tokens))
        ]

        color_list = [
            [0, 0, 0], [127, 123, 229], [195, 240, 251], [146, 223, 255],
            [243, 241, 230], [224, 190, 144], [178, 116, 75],
        ]
        self.color_tensor = (
            torch.tensor(color_list)[:, :, None, None]
            .repeat(1, 1, self.image_size, self.image_size)
        )

        patch_grid = (self.image_size // 14) ** 2

        # ── SINGLE ──────────────────────────────────────────────────────
        if self.gate == object_type.SINGLE:
            for i in range(self.shot):
                dilated = cv2.dilate(np.array(grounded_sam_masks[i][0]), self.kernel, iterations=1)
                thresh  = torch.tensor(dilated).reshape(1, 1, H, W)
                thresh  = F.interpolate(
                    thresh, size=self.image_size // 14,
                    mode="bilinear", align_corners=True,
                ).reshape(patch_grid)
                thresh[thresh > 0] = 1
                mask_idx = thresh.bool()

                self.normal_dino_part_patch_features[0].append(
                    self.normal_dino_patches[i][mask_idx]
                )
                for layer in range(len(self.normal_patch_tokens)):
                    if layer % 2 == 0:
                        continue
                    self.normal_clip_part_patch_features[layer][0].append(
                        self.normal_patch_tokens[layer][i][mask_idx]
                    )

            self.normal_dino_part_patch_features[0] = torch.cat(
                self.normal_dino_part_patch_features[0], dim=0
            )
            for layer in range(len(self.normal_patch_tokens)):
                if layer % 2 == 0:
                    continue
                self.normal_clip_part_patch_features[layer][0] = torch.cat(
                    self.normal_clip_part_patch_features[layer][0], dim=0
                )

        # ── MULTI ────────────────────────────────────────────────────────
        if self.gate == object_type.MULTI:
            _part_num = {
                "breakfast_box": [4], "screw_bag": [3],
                "splicing_connectors": [2], "pushpins": [3], "juice_bottle": [4],
            }
            _num_cluster = {k: 5 for k in _part_num}

            part_num_right = _part_num.get(self.class_name, [1])
            n_cluster      = _num_cluster.get(self.class_name, 2)

            cache_path = Path(self.heat_masks_root) / f"{self.class_name}_heat" / "train_features_sampled.pth"

            if re_seg:
                sampler = GreedyCoresetSampler(percentage=0.01, device=str(self.device))
                feats_list = []
                for img in dino_normal:
                    f0, _ = self.dino_net(img.unsqueeze(0))
                    f0    = f0.squeeze()
                    f0    = f0.reshape(f0.shape[0], -1).permute(1, 0)
                    feats_list.append(sampler.run(f0))
                train_feats = F.normalize(
                    torch.cat(feats_list, dim=0), dim=1
                ).cpu().numpy()

                part_num = -1
                retries  = 0
                while part_num not in part_num_right:
                    if retries >= _KMEANS_MAX_RETRIES:
                        break
                    retries += 1

                    km = KMeans(init="k-means++", n_clusters=n_cluster, n_init="auto")
                    km.fit(train_feats)
                    centers = torch.from_numpy(km.cluster_centers_)
                    tsf     = centers.to(self.device).unsqueeze(0).unsqueeze(0)
                    self.train_features_sampled = tsf.permute(0, 3, 1, 2)

                    for i, img in enumerate(dino_normal):
                        hm, hm_intra = get_heatmaps(
                            img.unsqueeze(0),
                            self.train_features_sampled,
                            self.dino_net,
                            self.color_tensor,
                            device=str(self.device),
                            use_crf=self.use_crf,
                        )
                        sp = self._heat_path(f"{self.class_name}_heat/train/{i}")
                        gs_m  = torch.stack(grounded_sam_masks[i])
                        cl_m  = torch.stack(split_masks_from_one_mask_torch(hm + 1))
                        hm_rf = assign_fine_to_coarse_torch(cl_m, gs_m).max(dim=0).values
                        cv2.imwrite(str(sp / "heatresult_refined.png"), hm_rf.detach().cpu().numpy())
                        _save_debug_image(img.unsqueeze(0), sp, hm_intra)
                        part_num = len(filter_bg_noise(self.heat_masks_root, self.class_name))
                        plt.clf()
                        plt.imshow(hm_rf.detach().cpu().numpy())
                        plt.savefig(str(sp / "masks_color.png"))
                        plt.close()

                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.train_features_sampled, cache_path)
            else:
                self.train_features_sampled = torch.load(cache_path, map_location=self.device)

            # Build per-component normal feature banks
            self.normal_component_feats = {k: [] for k in ["area", "color", "position", "clip_image", "dino_image", "geo"]}

            for i in range(self.shot):
                image_np = np.array(
                    Image.open(image_paths[i]).convert("RGB").resize((self.image_size, self.image_size))
                )
                mask_img_path = str(
                    Path(self.heat_masks_root) / f"{self.class_name}_heat" / "train" / str(i) / "heatresult_refined.png"
                )
                normal_masks_raw = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
                normal_masks_capm, normal_mask_idxs = split_masks_from_one_mask_with_bg(normal_masks_raw)
                normal_masks, _ = split_masks_from_one_mask(normal_masks_raw)

                k5 = np.ones((5, 5), np.uint8)
                normal_masks_capm = [cv2.dilate(m, self.kernel, iterations=1) for m in normal_masks_capm]
                normal_masks      = [cv2.dilate(m, k5,          iterations=1) for m in normal_masks]

                for j in range(len(normal_mask_idxs)):
                    thresh = torch.tensor(normal_masks_capm[j]).reshape(1, 1, H, W)
                    thresh = F.interpolate(
                        thresh, size=self.image_size // 14,
                        mode="bilinear", align_corners=True,
                    ).reshape(patch_grid)
                    if thresh.sum() < 1:
                        continue
                    thresh[thresh > 0] = 1
                    mask_idx = thresh.bool()
                    idx      = normal_mask_idxs[j]

                    self.normal_dino_part_patch_features[idx].append(
                        self.normal_dino_patches[i][mask_idx]
                    )
                    for layer in range(len(self.normal_patch_tokens)):
                        if layer % 2 == 0:
                            continue
                        self.normal_clip_part_patch_features[layer][idx].append(
                            self.normal_patch_tokens[layer][i][mask_idx]
                        )

                feats = self.component_feature_extractor.extract(image_np, normal_masks)
                for key in ["area", "color", "position", "clip_image", "dino_image"]:
                    self.normal_component_feats[key].append(feats[key])

            for idx in range(len(normal_mask_idxs)):
                pidx = normal_mask_idxs[idx]
                if not self.normal_dino_part_patch_features[pidx]:
                    continue
                self.normal_dino_part_patch_features[pidx] = torch.cat(
                    self.normal_dino_part_patch_features[pidx], dim=0
                )
                for layer in range(len(self.normal_patch_tokens)):
                    if layer % 2 == 0:
                        continue
                    self.normal_clip_part_patch_features[layer][pidx] = torch.cat(
                        self.normal_clip_part_patch_features[layer][pidx], dim=0
                    )

            for key in ["area", "color", "position", "clip_image", "dino_image"]:
                self.normal_component_feats[key] = torch.cat(
                    self.normal_component_feats[key], axis=0
                )
            self.normal_component_feats["clip_image"] = (
                self.normal_component_feats["clip_image"].transpose(0, 1)
            )
            for layer in range(self.normal_component_feats["clip_image"].shape[0]):
                self.normal_component_feats["clip_image"][layer] = self.cfa(
                    self.normal_component_feats["clip_image"][layer]
                )
            self.normal_component_feats["dino_image"] = self.cfa(
                self.normal_component_feats["dino_image"]
            )
            self.normal_component_feats["geo"] = torch.cat(
                [
                    self.normal_component_feats["area"],
                    self.normal_component_feats["color"],
                    self.normal_component_feats["position"],
                ],
                dim=1,
            )


# ---------------------------------------------------------------------------
# Supporting functions and classes
# ---------------------------------------------------------------------------

def calculate_iou_torch(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    return torch.sum((mask1 & mask2).float())


def assign_fine_to_coarse_torch(
    coarse_masks: torch.Tensor,
    fine_masks:   torch.Tensor,
) -> torch.Tensor:
    M, H, W = coarse_masks.shape
    N       = fine_masks.shape[0]

    mapping: dict[int, list] = {i: [] for i in range(M)}

    for fi in range(N):
        if N > 1:
            if fine_masks[fi][0, 0] and fine_masks[fi][H - 1, W - 1]:
                continue
            if fine_masks[fi][10, 10] and fine_masks[fi][H - 10, W - 10]:
                continue

        best_iou, best_ci = 0, -1
        for ci in range(M):
            iou = calculate_iou_torch(fine_masks[fi], coarse_masks[ci])
            if iou > best_iou:
                best_iou, best_ci = iou, ci

        if best_ci != -1:
            mapping[best_ci].append(fine_masks[fi])

    result = torch.zeros_like(coarse_masks)
    for ci, assigned in mapping.items():
        for fm in assigned:
            result[ci][fm > 0] = ci + 1

    return result


class CFA(nn.Module):
    """Cross-Feature Aggregation via normalised graph convolution."""

    def _similarity(self, x: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adj = self._similarity(x)
        adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return adj @ x
