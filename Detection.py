import numpy as np
from glob import glob
import cv2
import pytesseract
from commonfunctions import *
import cv2
import imutils
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.color import rgb2gray,rgb2hsv
from skimage.filters import  sobel_v,threshold_otsu,sobel
from skimage import feature
from skimage.transform import (hough_line, hough_line_peaks)
from skimage.morphology import binary_erosion, binary_dilation, binary_closing,skeletonize, thin
from commonfunctions import *
from scipy import ndimage
from difflib import SequenceMatcher
import os
import glob
import math
# Using tesseract to recognize characters
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'D:\\Programs\\Tesseract-OCR\\tesseract.exe'

class Detection:
    availableText = r'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    psmMode = 10
    tesseractConfig = r"-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 10"
    # Preprocess the image
    #  -> Get gray scale by multiplying red channel by 0.299, green channel by 0.587, blue channel by 0.114
    #  -> Remove noise using bilateral filter
    #  -> Apply histogram equalization on the filtered image


    @staticmethod 
    def preprocess(img):
        red_channel = img[:,:,0]
        green_channel = img[:,:,1]
        blue_channel = img[:,:,2]
        grey_image = img.copy()
        grey_image = (0.299*red_channel) + (0.587*green_channel) + (0.114*blue_channel) 
        img_filtered = cv2.bilateralFilter(grey_image.astype('uint8'), 11, 17, 17)
        img_filtered = img_filtered.astype('uint8')
        return img_filtered

    # Binarize the image
    @staticmethod 
    def binarizeImage(img):
        ret,img_binarized = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        return img_binarized

    # Inversely binarize the image
    @staticmethod 
    def inverseBinarizeImage(img):
        ret,img_binarized = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        return img_binarized

    # Morphological operations to be used later
    @staticmethod 
    def erosion(image, window_size):
        window = np.ones((window_size[0], window_size[1]), dtype=int)
        eroded_img = image[:, :]
        for i in range(image.shape[0] - window_size[0]):
            for j in range(image.shape[1] - window_size[1]):
                eroded_img[i, j] = np.all(np.logical_and(image[i:window_size[0] + i, j:window_size[1] + j], window))
        return eroded_img

    @staticmethod 
    def dilation(image, window_size): 
        window = np.ones((window_size[0], window_size[1]), dtype=int)
        dilated_img = image[:, :]
        for i in range(image.shape[0] - window_size[0]):
            for j in range(image.shape[1] - window_size[1]):
                dilated_img[i, j] = np.any(np.logical_and(image[i:window_size[0] + i, j:window_size[1] + j], window))
        return dilated_img

    @staticmethod 
    def getCharacter(index):
        index = int(index)
        characters = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"]
        character = characters[index]
        return character

    @staticmethod 
    def checkContours(test_img, img):
        contours1, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(test_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt1=contours1[0]
        cnt2=contours2[0]
        diff_factor = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
        return diff_factor

    @staticmethod 
    def detectLicense(image,idx):
        image = imutils.resize(image, width=500)
        image = Detection.preprocess(image)
        edged = cv2.Canny(image, 170, 200)
        edgedBinary = Detection.binarizeImage(edged)
        edgedBinary = edgedBinary.astype('uint8')
        cnts, new  = cv2.findContours(edgedBinary.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
        for c in cnts:
                areas=cv2.contourArea(c)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:  
                    x, y, w, h = cv2.boundingRect(c) 
                    new_img = image[y:y + h, x:x + w] 
                    return new_img

    # Segmentation of characters
    @staticmethod 
    def characterSegmentation(image) :
        
        dim0 = (20,40) # character shape dimension
        dim1 = (44,24) # character full image dimension
        dim2 = (333, 75)
        window_size = [3,3]
        lower_width, upper_width = 6,70
        lower_height, upper_height = 16, image.shape[1]
        target_contours = []
        img_res = []
        index_list = []
        contour_list = []
        
        img = cv2.resize(image, dim2)
        _, img_binarized = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        img_binarized = Detection.dilation(img_binarized, window_size)
        img_binarized = Detection.erosion(img_binarized, window_size)

        img_binarized=img_binarized.astype('uint8')
        img_binarized=255*img_binarized
            
        contours, _ = cv2.findContours(img_binarized.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
        
        for contour in contours :
            x, y, width, height = cv2.boundingRect(contour)   
            if (width >= lower_width and width <= upper_width) and (height >= lower_height and height <= upper_height):     
                    contour_list.append(x)
                    char = img_binarized[y:y+height, x:x+width]
                    char = cv2.resize(char, dim0)
                    char = 255 - char
                    char_full_img = np.zeros(dim1)
                    char_full_img[2:42, 2:22] = char
                    img_res.append(char_full_img)
            
        indices = sorted(range(len(contour_list)), key=lambda k: contour_list[k])
        
        for index in indices:
            index_list.append(img_res[index])
        img_res = np.array(index_list)
        return img_res

    @staticmethod 
    def compareCharacters(test_img, img):  
        count = 0
        length = img.shape[0]
        width = img.shape[1]
        img = img.astype('uint8')
        test_img = test_img.astype('uint8')
    #     diff_factor = checkContours(test_img, img)
        for i in range(length):
            for j in range(width):
                if(img[i, j]==test_img[i, j]): 
                    count+=1
        return count

    # Using direct comparison with input set to recognize characters
    @staticmethod 
    def characterRecognition(test_img):     
        count_match_list = []  
        num_pics_per_char = 10
        i = 0
        dim = (int(24), int(44))
        sums = np.zeros(36)
        test_img = Detection.binarizeImage(test_img) 
        for filename in sorted(glob.glob('./LPR Digits/*.jpg')): 
            img = cv2.imread(filename)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            img = Detection.preprocess(img)
            img = Detection.inverseBinarizeImage(img)
            count_match = Detection.compareCharacters(test_img,img)
            sums[i // 10] += count_match
            i += 1
            count_match_list.append(count_match)   
        count_match_array = np.array(count_match_list)
        index = (np.argmax(count_match_array))//num_pics_per_char
        return Detection.getCharacter(index)

    @staticmethod 
    def tesseractCharacterRecognition(image):
        # Converting the incoming binary image to RGB image, as tesseract takes only RGB images
        new_p = Image.fromarray(image)
        if new_p.mode != 'RGB':
            new_p = new_p.convert('RGB')
        text = pytesseract.image_to_string(new_p, config = Detection.tesseractConfig, lang = "eng").split()
        if len(text) == 1 and text[0] in Detection.availableText:
            return text[0]
        characters = ""
        for c in text:
            characters += c
        return characters

    @staticmethod 
    def singleTesting(carNumber):
        # Return image
        image=cv2.imread(f'./dataset/images/{carNumber}')

        # Return img_cropped
        img_cropped= Detection.detectLicense(image,0)

        resultimg = img_cropped
        imag=Detection.characterSegmentation(resultimg)
        final=[]
        for i in range(len(imag)):
            summ=np.sum(imag[i])/(imag[i].shape[0]*imag[i].shape[1])
            if(summ>50):
                final.append(imag[i])
        # Return final

        # Characters Recognition =>
        PlateNumber = ""
        PlateNumberList = []
        failedCharactersIndices = []

        # Recognize the characters using tesseract
        i = 0
        for Char in final:
            tessChar = Detection.tesseractCharacterRecognition(Char)
            if tessChar == "":
                failedCharactersIndices.append(i)
            PlateNumberList.append(tessChar)
            i += 1

        # Recognize the characters by comparing the character with the dataset
        i = 0
        for Char in final:
            if PlateNumberList[i] == "":
                PlateNumberList[i] = Detection.characterRecognition(Char)
            i += 1

        # Append characters in one string
        for digit in PlateNumberList:
            PlateNumber += digit
        return img_cropped, final, PlateNumber
        # Return PlateNumber