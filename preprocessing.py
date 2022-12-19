import cv2
import numpy as np

class Preprocessing:

    # Gets a gray scale image with each of the RGB channels having different weights
    @staticmethod
    def grayScale(image):
        grayImage = np.zeros(image.shape)
        # These numbers follow the paper given (page 6)
        grayImage = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        return grayImage

    # Remove the noise from the image using the bilateralFilter
    @staticmethod
    def removeNoise(image):
        # Takes the image, the diameter for filteration 'd', sigma color, and sigma coordinate for color and coordinate spaces
        # If a value is larger than the given parameters for the sigmas,
        # the colors within the neighbourhood are mixed together
        # The sigma coordinate is proportional to the diameter of filteration
        return cv2.bilateralFilter(image.astype('uint8'), 10, 20, 20)

    # Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied
    @staticmethod
    def equilizeHistogram(image):
        # The declaration of CLAHE having a threshold for contrast limiting then applied to the image
        # which is divided into 8x8 tiles
        clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
        equalizedImage = clahe.apply(image.astype('uint8'))
        return equalizedImage

    # All preprocessing operations collected in one function
    @staticmethod
    def preprocessPhoto(image):
        grayImage = Preprocessing.grayScale(image)
        noiseFreeImage = Preprocessing.removeNoise(grayImage)
        equilizedImage = Preprocessing.equilizeHistogram(noiseFreeImage)
        return equilizedImage, grayImage