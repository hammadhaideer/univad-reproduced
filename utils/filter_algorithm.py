import glob
from pathlib import Path

import cv2
import numpy as np

_HEATMAP_SCALE = 128
_HEATMAP_THRESHOLD = _HEATMAP_SCALE // 2
_CORNER_SIZE = 20


def filter_bg_noise(sourcepath: str, classname: str) -> list:
    train_file_path = Path(sourcepath) / f"{classname}_heat" / "train"
    trainfiles = sorted(
        glob.glob(str(train_file_path / "*")),
        key=lambda x: int(Path(x).name),
    )
    img0_path = Path(trainfiles[0])
    reserve_list = []

    seg_img_list = sorted(
        glob.glob(str(img0_path / "heatresult[0-9].jpg"))
    )

    for i, imgpath in enumerate(seg_img_list):
        gray_img = cv2.imread(imgpath, 0)
        H, W = gray_img.shape

        gray_cal_otsu = gray_img[10:H - 10, 10:W - 10]
        ret, _ = cv2.threshold(
            gray_cal_otsu, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        _, thresh2 = cv2.threshold(gray_img, ret, 1, cv2.THRESH_BINARY)

        corner1 = np.zeros_like(thresh2)
        corner1[0:_CORNER_SIZE, 0:_CORNER_SIZE] = 1

        corner2 = np.zeros_like(thresh2)
        corner2[H - _CORNER_SIZE:H, 0:_CORNER_SIZE] = 1

        corner3 = np.zeros_like(thresh2)
        corner3[0:_CORNER_SIZE, W - _CORNER_SIZE:W] = 1

        corner4 = np.zeros_like(thresh2)
        corner4[H - _CORNER_SIZE:H, W - _CORNER_SIZE:W] = 1

        ex = (
            (corner1 * thresh2).max()
            + (corner2 * thresh2).max()
            + (corner3 * thresh2).max()
            + (corner4 * thresh2).max()
        )

        kernel_size = (11, 11)
        gray_img = cv2.blur(gray_img, kernel_size)
        maxvalue = gray_img.max()

        if maxvalue > _HEATMAP_THRESHOLD and ex < 3:
            reserve_list.append(i)

    return reserve_list