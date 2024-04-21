import cv2
import numpy as np


class ImageProcessor:
    def __init__(self, image):
        self.image = image

    def validate_contour(self, contour):
        """Validate that the segmentated contour is a triangle"""
        # approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.06 * peri, True)

        # if the polygon has 3 vertices, it's a triangle
        if len(approx) == 3:
            # find centroid of the contour
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            return (cx, cy)

        return None

    def preprocess_image(self, image):
        # denoising image
        image = cv2.GaussianBlur(image, (7, 7), 0)
        image = cv2.medianBlur(image, 13)

        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2.8)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (3, 3), iterations=2)
        #thresh = cv2.erode(thresh, (3, 3), iterations=3)
        #thresh = cv2.dilate(thresh, (5, 5), iterations=8)

        return thresh

    def segmentate_objects(self):
        # preprocess the image by each channel separately
        preprocessed_image = None
        for ch in cv2.split(self.image):
            if preprocessed_image is None:
                preprocessed_image = np.zeros_like(ch)
            preprocessed_channel = self.preprocess_image(ch)
            preprocessed_image = cv2.bitwise_or(preprocessed_image, preprocessed_channel)

        contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # filter contours which are triangles
        centroids_validated = []
        contours_validated = []

        for contour in contours:
            dot_coords = self.validate_contour(contour)
            if dot_coords:
                centroids_validated.append(dot_coords)
                contours_validated.append(contour)
        return centroids_validated, contours_validated
