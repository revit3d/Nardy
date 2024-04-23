import os
from typing import Generator, List

import numpy as np
import cv2
from cv2.typing import MatLike

from tiles_segmentation import TileExtractor, Contour
from tiles_classification import TileClassifier


class ImageProcessor:
    def __init__(self):
        pass

    def extract_tile_images(self,
                            image: MatLike,
                            tile_contours: List[Contour]) -> Generator[MatLike]:
        """Extract images with tiles standing upright"""
        for contour in tile_contours:
            # find the <<bottom>> side of the triangle
            a, b = sorted(contour.vertices[:, 0], key=lambda x: x[1], reverse=True)[:2]
            x, y = a - b

            # calculate angle and center of rotation
            angle = np.degrees(np.arctan2(y, x))
            angle = angle if angle < 90 else angle + 180
            center = (contour.centroid.x, contour.centroid.y)

            # rotate image so the triangle will stand upright
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            rotated_contour = cv2.transform(contour.vertices, M)

            # crop a square image of a new tile
            x, y, w, h = cv2.boundingRect(rotated_contour)
            size = max(w, h)
            cropped = rotated_image[y:y + size, x:x + size]
            yield cropped

    def process_image(self, image: MatLike, file_path: os.PathLike) -> None:
        """
        Process image and count number of triomino tiles on the image.\\
        Then classify each tile by the dots on its vertices.
        ------
        Parameters:
        ------
        - image: image suitable for requirements
        - file_path: path to the file with processing results
        """
        extractor = TileExtractor()
        contours = extractor.find_objects(image)

        classifier = TileClassifier()
        preds = []
        for tile_img in self.extract_tile_images(image, contours):
            preds.append(classifier.predict(tile_img))

        assert len(contours) == len(preds)
        with open(file_path, 'w') as fout:
            fout.write(f'{len(contours)}\n')
            for contour, pred in zip(contours, preds):
                c = contour.centroid
                cls = ', '.join(map(str, pred))
                fout.write(f'{c.x}, {c.y}; ' + cls + '\n')
