import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import *

def preprocess(path):
    fingerprint = cv.imread(path, cv.IMREAD_GRAYSCALE)
    gx, gy = cv.Sobel(fingerprint, cv.CV_32F, 1, 0), cv.Sobel(fingerprint, cv.CV_32F, 0, 1) # Calculate the local gradient (using Sobel filters)
    gx2, gy2 = gx**2, gy**2 # Calculate the magnitude of the gradient for each pixel
    gm = np.sqrt(gx2 + gy2)
    sum_gm = cv.boxFilter(gm, -1, (25, 25), normalize = False) # Integral over a square window
    thr = sum_gm.max() * 0.2 # Use a simple threshold for segmenting the fingerprint pattern
    mask = cv.threshold(sum_gm, thr, 255, cv.THRESH_BINARY)[1].astype(np.uint8)
    enhanced = mask & np.clip(fingerprint, 0, 255).astype(np.uint8)
    cv.imwrite('enhanced.jpg',enhanced)
    fingerprint[mask<1] = 255
    cv.imwrite('finger.jpg',fingerprint)
    cv.imwrite('mask.jpg',mask)
    _, ridge_lines = cv.threshold(fingerprint, 0, 255,cv.THRESH_OTSU)
    cv.imwrite('bina.jpg',ridge_lines)


if __name__=='__main__':
    path = '0.bmp'
    preprocess(path)


