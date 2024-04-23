from __future__ import annotations

import math
from typing import Literal, Optional, Iterable, Tuple, List
from dataclasses import dataclass

import cv2
import numpy as np
from cv2.typing import MatLike


@dataclass
class Point:
    x: int
    y: int

    def dist(self, other: Point) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class Contour:
    centroid: Point
    vertices: MatLike

    def is_equilaterial(self, thres: float) -> bool:
        if len(self.vertices) != 3:
            raise ValueError('Equilaterial check can be used only for triangle contours')

        sides = []
        for i in range(3):
            for j in range(i + 1, 3):
                pt1 = Point(*self.vertices[i][0])
                pt2 = Point(*self.vertices[j][0])
                sides.append(pt1.dist(pt2))
        assert len(sides) == 3
        return (max(sides) - min(sides)) / max(sides) < thres


class TileExtractor:
    def __init__(self,
                 opening_kernel: Tuple[int, int] = (3, 3),
                 gblur_kernel: Tuple[int, int] = (5, 5),
                 medblur_kernel: int = 15,
                 approx_thres: float = 0.05,
                 equilaterial_thres: float = 0.15,
                 area_min: int = 2000,
                 area_max: int = 7000,
                 min_dist_between_centr: int = 20):
        """
        Processes the image, segmentating triomino tiles on the image\\
        and classifying it
        ------
        Parameters:
        ------
        - opening_kernel: kernel of opening applied after thresholding
        - gblur_kernel: gaussian blur kernel
        - medblur_kernel: median blur kernel
        - approx_thres: threshold for approximating contour by triangle
        - equilaterial_thres: threshold for approximating equilaterial triangles
        - area_min: min area of triomino to segmentate (used for filtering out noise)
        - area_max: max area of triomino to segmentate (used for filtering out noise)
        - min_dist_between_centr: minimal distance between two centroids of triomino tiles
        """
        self.opening_kernel = opening_kernel
        self.gblur_kernel = gblur_kernel
        self.medblur_kernel = medblur_kernel

        self.approx_thres = approx_thres
        self.equilaterial_thres = equilaterial_thres

        self.area_min = area_min
        self.area_max = area_max
        self.min_dist_between_centr = min_dist_between_centr

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
        approx = cv2.approxPolyDP(contour, self.approx_thres * peri, True)

        # check the approximated polygon has 3 vertices (triangle)
        if len(approx) != 3:
            return None

        M = cv2.moments(approx)

        # filter noise contours by area
        if self.area_min > M['m00'] or M['m00'] > self.area_max:
            return None

        centroid = Point(x=int(M['m10'] / M['m00']), y=int(M['m01'] / M['m00']))
        new_contour = Contour(centroid=centroid, vertices=approx)

        # filter noise contours which are not equilaterial
        if not new_contour.is_equilaterial(self.equilaterial_thres):
            return None

        return new_contour

    def smoothen(self, image: MatLike) -> MatLike:
        image = cv2.GaussianBlur(image, self.gblur_kernel, 0)
        image = cv2.medianBlur(image, self.medblur_kernel)
        return image

    def process_adaptive(self, image: MatLike) -> MatLike:
        thresh = cv2.adaptiveThreshold(image, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # remove excessive noise
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.opening_kernel, iterations=1)
        return thresh

    def process_otsu(self, image: MatLike) -> MatLike:
        # uniform background illumination for Otsu thresholding
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
        image = cv2.divide(image, bg, scale=255)

        # apply otsu thresholding
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # remove excessive noise
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.opening_kernel, iterations=2)
        return thresh

    def process_image(self, image: MatLike, how: Literal['otsu', 'adaptive'] = 'otsu') -> MatLike:
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
        if len(contours) == 0:
            return []

        unique = [contours[0], ]
        for contour in contours[1:]:
            min_dist_to_uni = min([contour.centroid.dist(other.centroid) for other in unique])
            if min_dist_to_uni > self.min_dist_between_centr:
                unique.append(contour)
        return unique

    def find_objects(self, image: MatLike) -> List[Contour]:
        thres_image_otsu = self.process_image(image, 'otsu')
        thres_image_ada = self.process_image(image, 'adaptive')

        contours_otsu, _ = cv2.findContours(thres_image_otsu,
                                            cv2.RETR_LIST,
                                            cv2.CHAIN_APPROX_SIMPLE)
        contours_ada, _ = cv2.findContours(thres_image_ada,
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)

        contours_otsu = self.process_contours(contours_otsu)
        contours_ada = self.process_contours(contours_ada)

        unique_contours = self.find_unique_contours(contours_otsu + contours_ada)

        return unique_contours
