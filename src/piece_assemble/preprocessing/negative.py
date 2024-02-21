from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from skimage.filters import median, threshold_otsu
from skimage.measure import label, regionprops

from piece_assemble.preprocessing import np_to_pil, pil_to_np
from piece_assemble.preprocessing.base import PieceExtractorBase

if TYPE_CHECKING:
    from PIL.Image import Image


class NegativePieceExtractor(PieceExtractorBase):
    def __call__(self, img: Image) -> tuple[Image, np.ndarray]:
        img_gray = img.convert("L")
        img_np = pil_to_np(img_gray)
        img_bin = self.binarize(img_np)
        img_mask, bbox = self.get_piece_mask(img_bin)
        img_rgb_np = pil_to_np(img)
        img_piece = np.where(np.expand_dims(img_mask, 2), img_rgb_np, 1)[
            bbox[0] : bbox[2], bbox[1] : bbox[3]
        ]
        piece_mask = img_mask[bbox[0] : bbox[2], bbox[1] : bbox[3]]

        return np_to_pil(img_piece), piece_mask

    def binarize(
        self, img: np.ndarray[int], bg_relative_var: float = 1.1
    ) -> np.ndarray[bool]:
        # This method is designed for images with single colored background
        img_median = median(img, footprint=np.ones((11, 11)))

        # First rough binarization
        thr = threshold_otsu(img_median)

        background_values = img_median[np.where(img_median > thr)]
        background_median = np.median(background_values)

        std = np.std(background_values)
        tol = bg_relative_var * std

        return (img_median < background_median - tol) | (
            img_median > background_median + tol
        )

    def get_piece_mask(
        self, img_bin: np.ndarray[bool]
    ) -> tuple[np.ndarray[bool], tuple[int, int, int, int]]:
        labels = label(img_bin)
        rp = regionprops(labels)

        areas = [region["area"] for region in rp]
        region = rp[np.argmax(areas)]
        mask_crop = region["image_filled"]
        bbox = region["bbox"]
        img_mask = np.zeros(img_bin.shape)
        img_mask[bbox[0] : bbox[2], bbox[1] : bbox[3]] = mask_crop

        return img_mask, bbox
