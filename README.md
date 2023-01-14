# License Plate Recognition

## Introduction

> This is the project of the [Image Processing and Computer Vision] academic elective course `CMPN446` in Cairo University - Faculty of Engineering - Credit Hours System - Communication and Computer Engineering Program
>
> This project is about applying image processing techniques to localize the license plate in an image and apply OCR to get the license plate number.

***

## Used Technologies

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"> <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"> <img src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white"> <img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white"> <img src="https://img.shields.io/badge/scikit--image-%23F7931E.svg?style=for-the-badge&logo=scipy&logoColor=white">

***

## Pipeline

> Our implementation is strongly inspired by the research paper of ([A New Approach for License Plate Detection and Localization Between Reality and Applicability](https://www.researchgate.net/publication/283758434_A_New_Approach_for_License_Plate_Detection_and_Localization_Between_Reality_and_Applicability)). However, we have read other papers and started to mix between the approaches that we have found until we decided on the following pipeline.

### Preprocessing:

1. Convert image to grayscale
2. Remove noise by applying bilateral filter
3. Contrast enhancement using Contrast Limited Adaptive Histogram Equalization

### Plate Detection:

1. Vertical edge detection using sobel
2. Image Binarization
3. ROI mask to divide the image into regions of interests according to variance
4. Filter regions according to their sizes
5. Harris corner detection and dilation on remaining regions
6. Weighting to remaining regions according to closeness of corners
7. Choosing region with highest weight
8. Getting contours of the best region to detect the bounding rectangle of the plate

### Character Recognition:

1. Adjusting the phase of the plate
2. Character segmentation
3. Binarization and morphological operations to prepare characters for OCR
4. OCR using `pytesseract` in addition to pattern matching

***

## Team Members

1. [Esraa Hamed](https://github.com/esraa-hamed)
2. [George Joseph](https://github.com/George-Joseph99)
3. [Mohamed Ahmed Zaki](https://github.com/mohammedzaki7)
4. [Khaled El-Lethy](https://github.com/ellethykhaled)

***

