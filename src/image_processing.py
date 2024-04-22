from __future__ import annotations

import math
from typing import Literal, Optional, Iterable, List
from dataclasses import dataclass

import cv2
import numpy as np
from cv2.typing import MatLike


@dataclass
class Centroid:
    x: int
    y: int

    def dist(self, other: Centroid) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class Contour:
    centroid: Centroid
    vertices: MatLike


class ImageProcessor:
    def __init__(self):
        pass

    def process_contours(self, contours: Iterable[MatLike]) -> List[Contour]:
        """Approximate and filter out noise contours"""
        contours_validated = []

        for contour in contours:
            approx = self.validate_contour(contour)
            if approx:
                contours_validated.append(approx)

        return contours_validated

    def validate_contour(self, contour: MatLike) -> Optional[Contour]:
        """
        Validate that the segmentated contour can be approximated by a triangle \\
        and return its centroid and approximated triangle or None otherwise
        """
        # approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)

        # if the polygon has 3 vertices, it's a triangle
        if len(approx) == 3:
            # find centroid of the contour
            M = cv2.moments(approx)

            # filter noise contours
            if 2000 < M['m00'] < 7000:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return Contour(centroid=Centroid(cx, cy), vertices=approx)

    def smoothen(self, image: MatLike) -> MatLike:
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = cv2.medianBlur(image, 15)
        return image

    def process_adaptive(self, image: MatLike) -> MatLike:
        thresh = cv2.adaptiveThreshold(image, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # remove excessive noise
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (3, 3), iterations=1)
        return thresh

    def process_otsu(self, image: MatLike) -> MatLike:
        # uniform background illumination for Otsu thresholding
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
        image = cv2.divide(image, bg, scale=255)

        # apply otsu thresholding
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # remove excessive noise
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (3, 3), iterations=2)
        return thresh

    def process_image(self, image: MatLike, how: Literal['otsu', 'adaptive']='otsu') -> MatLike:
        """Preprocess the image by each channel separately"""
        process_func = {
            'otsu': self.process_otsu,
            'adaptive': self.process_adaptive,
        }

        channels = cv2.split(image)
        processed_image = np.zeros_like(channels[0])
        for ch in channels:
            preprocessed_channel = self.smoothen(ch)
            processed_channel = process_func[how](preprocessed_channel)
            processed_image = cv2.bitwise_or(processed_image, processed_channel)
        return processed_image
    
    def find_unique_contours(self, contours: List[Contour]) -> List[Contour]:
        """Filter out overlapping contours"""
        unique = [contours[0], ]
        for contour in contours[1:]:
            min_dist_to_uni = min([contour.centroid.dist(other.centroid) for other in unique])
            if min_dist_to_uni > 100:
                unique.append(contour)
        return unique

    def segmentate_objects(self, image: MatLike) -> List[Contour]:
        thres_image_otsu = self.process_image(image, 'otsu')
        thres_image_ada = self.process_image(image, 'adaptive')

        contours_otsu, _ = cv2.findContours(thres_image_otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_ada, _ = cv2.findContours(thres_image_ada, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contours_otsu = self.process_contours(contours_otsu)
        contours_ada = self.process_contours(contours_ada)

        unique_contours = self.find_unique_contours(contours_otsu + contours_ada)

        return unique_contours
