#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF

try:
    import pydensecrf.densecrf as dcrf
    import pydensecrf.utils as utils
    _CRF_AVAILABLE = True
except ImportError:
    _CRF_AVAILABLE = False

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

MAX_ITER = 2


def _unnorm(t: torch.Tensor) -> torch.Tensor:
    return t.cpu() * _IMAGENET_STD + _IMAGENET_MEAN


def dense_crf(image_tensor: torch.Tensor, output_logits: torch.Tensor) -> np.ndarray:
    if not _CRF_AVAILABLE:
        raise RuntimeError(
            "pydensecrf is not installed. Use --no_crf flag to skip CRF post-processing."
        )

    image = np.array(VF.to_pil_image(_unnorm(image_tensor)))[:, :, ::-1]
    H, W  = image.shape[:2]
    image = np.ascontiguousarray(image)

    output_logits = F.interpolate(
        output_logits.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    c, h, w = output_probs.shape

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=1, compat=3)
    d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=image, compat=4)

    Q = d.inference(MAX_ITER)
    return np.array(Q).reshape((c, h, w))
