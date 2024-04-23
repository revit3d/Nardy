from typing import Tuple

import numpy as np
import cv2
from cv2.typing import MatLike


class TileClassifier:
    def __init__(self):
        pass

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
        clusters = [image[:size // 2],
                    image[size // 2:, :size // 2],
                    image[size // 2:, size // 2:]]
        
        cls = []
        for img in clusters:
            pass
