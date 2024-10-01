from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from skimage.filters import median, threshold_otsu
from skimage.measure import label, regionprops
from skimage.transform import rescale

from image import np_to_pil, pil_to_np
from piece_assemble.preprocessing.base import PieceExtractorBase
from piece_assemble.preprocessing.common import get_resize_shape

if TYPE_CHECKING:
    from PIL.Image import Image


class NegativePieceExtractor(PieceExtractorBase):
    def __init__(
        self,
        background_var: float = 1.5,
        fill_holes: bool = True,
        max_image_size: int | None = None,
    ) -> None:
        """Initialize NegativePieceExtractor.

        Parameters
        ----------
        background_var
            Number describing variance of the background. If the number is close,
            the pixel value must be really close to the estimated background median to
            be considered background.
            Set this value low (~1) if there is a low contrast between the piece and
            the background and the color of the background doesn't change much.
            Set this value high (>2) if the contrast between the piece and the
            background is high or the color of the background varies (e.g. gradient).
        fill_holes
            Whether the holes in the piece mask are allowed.
            It is not advised to change the default value `True` unless you truly expect
            pieces with holes, as it eliminates the noise in the segmentation.
        max_image_size
            If set, the image might be downsized to the maximal allowed size for more
            time efficient computation.
            The output image will be upscaled again to match the original size, but the
            precision of the segmentation might be worse.

        """
        super().__init__()

        self.background_var = background_var
        self.fill_holes = fill_holes
        self.max_image_size = max_image_size

    def __call__(self, img: Image) -> tuple[Image, np.ndarray]:
        #
        original_shape = img.size
        if self.max_image_size:
            new_shape = get_resize_shape(original_shape, self.max_image_size)
            scale = new_shape[0] / original_shape[0]
        else:
            scale = 1

        if scale != 1:
            img = img.resize(new_shape)

        img_gray = img.convert("L")
        img_np = pil_to_np(img_gray)
        img_bin = self.binarize(img_np)
        img_mask, bbox = self.get_piece_mask(img_bin)
        img_rgb_np = pil_to_np(img)
        img_piece = np.where(np.expand_dims(img_mask, 2), img_rgb_np, 1)[
            bbox[0] : bbox[2], bbox[1] : bbox[3]
        ]
        piece_mask = img_mask[bbox[0] : bbox[2], bbox[1] : bbox[3]]

        if scale != 1:
            img_piece = rescale(img_piece, 1 / scale, channel_axis=2)
            piece_mask = rescale(piece_mask.astype(bool), 1 / scale)

        return np_to_pil(img_piece), piece_mask

    def _get_median_footprint(self, shape: tuple[int, int]) -> tuple[int, int]:
        """Return the footprint of the median filter based on the image shape"""
        kernel_size = (np.max(shape) // 400) * 2 + 1
        kernel_size = np.clip(kernel_size, 3, 15)
        return np.ones((kernel_size, kernel_size))

    def binarize(self, img: np.ndarray[int]) -> np.ndarray[bool]:
        # This method is designed for images with single colored background
        img_median = median(img, footprint=self._get_median_footprint(img.shape))

        # First rough binarization
        thr = threshold_otsu(img_median)

        background_values = img_median[np.where(img_median > thr)]
        background_median = np.median(background_values)

        std = np.std(background_values)
        tol = self.background_var * std

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
        mask_crop = region["image_filled"] if self.fill_holes else region["image"]
        bbox = region["bbox"]
        img_mask = np.zeros(img_bin.shape)
        img_mask[bbox[0] : bbox[2], bbox[1] : bbox[3]] = mask_crop

        return img_mask, bbox
