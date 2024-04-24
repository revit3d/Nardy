from typing import Tuple

import numpy as np
import cv2
from cv2.typing import MatLike

from .tiles_segmentation import Point


class TileClassifier:
    def __init__(self,
                 center_dist_thres: float = 4,
                 min_dot_r: float = 2.5,
                 max_dot_r: float = 6):
        self.center_dist_thres = center_dist_thres
        self.min_dot_r = min_dot_r
        self.max_dot_r = max_dot_r

    def predict(self, image: MatLike) -> Tuple[int, int, int]:
        """
        Predict class (tuple of three digits) of triomino tile.\\
        The image has to represent a tile of triomino standing\\
        in upright position. The image has to have 1:1 ratio.
        ------
        Parameters:
        ------
        - image: image suitable for requirements
        ------
        Returns:
        ------
        Three integers: number of dots in each vertex of triomino tile
        """
        # crop the image into three distinct images with different clusters of dots
        size = image.shape[0]
        clusters = [image[:size // 2, size // 4:size - size // 4],
                    image[size // 2:, :size // 2],
                    image[size // 2:, size // 2:]]

        # adaptive enlightening of the image
        enlighten_coef = 100 / np.mean(image)
        image = (image / 255) * enlighten_coef
        image[image > 1] = 1
        image = (image * 255).astype(np.uint8)

        cls = []
        for img in clusters:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            chs = list(cv2.split(img))
            chs[1] = 255 - chs[1]
            thresh = np.zeros_like(chs[0])
            for ch in chs:
                ch = cv2.medianBlur(ch, 3)
                t = cv2.Canny(ch, 70, 140)
                thresh = cv2.bitwise_or(thresh, t)

            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            segmentated_dots = []
            for contour in contours:
                (x, y), r = cv2.minEnclosingCircle(contour)
                center = Point(int(x), int(y))
                dsts = [center.dist(pt) for pt in segmentated_dots]
                is_unique = len(dsts) == 0 or min(dsts) > self.center_dist_thres
                if (self.min_dot_r < r < self.max_dot_r and is_unique):
                    segmentated_dots.append(center)
            cls.append(len(segmentated_dots))
        return tuple(cls)
