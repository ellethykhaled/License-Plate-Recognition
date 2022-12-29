import cv2
import numpy as np
import math
from skimage import io
from skimage.filters import  sobel_v,threshold_otsu,sobel
from scipy import ndimage
import pytesseract
from commonfunctions import *

class LicenceDetection:
    harris_corner = False
    increase_number = 20
    debug = False

    # Gets the map containing vertical edges using built-in sobel filter
    @staticmethod
    def detectVerticalEdges(image):
        verticalEdgesDetectedImage = np.abs(sobel_v(image))
        return verticalEdgesDetectedImage

    # This function gets the map containing weighted edges (strong and weak)
    @staticmethod
    def getWeightedEdges(verticalEdgesMap):
        imageMean = np.mean(verticalEdgesMap)
        # The threshold used in image binarization 
        threshVx = imageMean + 3.5 * imageMean

        prevX = 0
        prevY = 0
        dist = 0
        weightedEdges = np.copy(verticalEdgesMap)
        # Looping on each and every pixel
        for x in range(verticalEdgesMap.shape[1]):
            for y in range(verticalEdgesMap.shape[0]):
                if verticalEdgesMap[y, x] > threshVx:
                    # In case the current pixel's intensity is greater than the threshold
                    # calculate the distance between it and the previous edge
                    dist = math.sqrt((prevX - x) ** 2 + (prevY - y) ** 2)
                    if dist < 15:
                        # Count it as a strong edge if the distance is less than 15
                        weightedEdges[y, x] = 1
                    else:
                        # Otherwise, count it as a weak edge
                        weightedEdges[y, x] = 0.5
                else:
                    # Otherwise, reset the pixel as a non-edge
                    weightedEdges[y, x] = 0
        return weightedEdges

    @staticmethod
    def initialRoiRegion(weightedEdges, grayImage):
        # Calculate the variance for each vertical edge within the map of weighted edges
        rowVariance = np.var(weightedEdges, axis = 1)
        # Get the threshold (as in page 7)
        threshVarMax = max(rowVariance) / 3
        roiImage = np.zeros(weightedEdges.shape)
        # Anything that is greater than the threshold calculated is taken into consideration,
        # otherwise, is kept as 0
        roiImage[rowVariance > threshVarMax, :] = grayImage[rowVariance > threshVarMax, :]

        # An array of sums for each region
        roiSum = np.sum(roiImage, axis = 1)
        # Iterators to keep track of the start and end of the roi regions
        roiStart = 0
        roiEnd = 0
        # A list to keep track of the start and end of the roi regions
        roiRegions = []

        inRegion = False
        for i in range(len(roiSum)):
            if roiSum[i] != 0 and inRegion == False:
                # If the list is not empty and the roiEnd added last subtracted
                # from the main iterator (i) is less than 10, remove the last
                # ROI and set the roiStart iterator with the removed ROI 
                if len(roiRegions) != 0 and i - roiRegions[-1][1] < 10:
                    roiStart, _ = roiRegions.pop()
                else:
                    roiStart = i
                inRegion = True
            if roiSum[i] == 0 and inRegion == True:
                roiEnd = i - 1
                inRegion = False
                
                # ROI with height less than 15 are not considered
                if roiEnd - roiStart > 15:
                    roiRegions.append([roiStart, roiEnd])

        # Append the saved iterators to the roiRegions list in case the list is empty
        # or the iterators are not in the list (written like that to prevent runtime error)
        if len(roiRegions) == 0 or roiRegions[-1][0] != roiStart:
            roiRegions.append([roiStart,roiEnd])

        filteredRegions = []
        for region in roiRegions:
            # Take only the regions within the given sizes
            if region[1] - region[0] > 10 and region[1] - region[0] < grayImage.shape[0] / 4.5:
                filteredRegions.append(region)
        
        return filteredRegions

    @staticmethod
    def getBestRegion(roiRegions, weightedEdges, image):

        # Return the whole image
        if len(roiRegions) == 0 : return [0, image.shape[0]]
        
        # Return the single roi extracted
        if len(roiRegions) == 1 : return roiRegions[0]
        
        bestRegionIndex = 0
        bestWeight = 0
        
        if LicenceDetection.harris_corner:
            # Looping on all ROIs extracted
            for i in range(len(roiRegions)):
                # Get the ROI as a 2D colored image (with 3 channels)
                regionImage = image[roiRegions[i][0] : roiRegions[i][1], :]
                gray = np.float32(regionImage)
                # The Harris corner detector takes the image,
                # block size, aperture parameter of the Sobel derivative and 'k' free parameter
                dst = cv2.cornerHarris(gray, 4, 7, 0.2)
                # Dilation is used to remove unimportant corners
                dst = cv2.dilate(dst, None)
                
                # A map containing important corners
                testImage = np.zeros(regionImage.shape)
                testImage[dst > 0.25 * dst.max()] = 255
           
                regionWeight = 0
                # Ignoring extreme left and right parts of the image
                start = testImage.shape[1] // 4
                end = testImage.shape[1] - start
                for k in range(testImage.shape[0]):
                    prevEdge = 0
                    for j in range(0,testImage.shape[1]):
                        # In case of corner point
                        if testImage[k][j] == 255:
                            weightFactor = 1 / 50
                            # Set the weight factor by 50 if not an extreme corner (not left or right)
                            if j > start and j < end:
                                weightFactor = 50
                            if prevEdge == 0:
                                regionWeight += 1 * weightFactor
                            else:
                                dist = np.abs(prevEdge - j)
                                regionWeight += 1 / np.exp(dist) * weightFactor
                            prevEdge = j
                # Divide the region weight by the height of the ROI for fair comparison
                regionWeight /= (roiRegions[i][1] - roiRegions[i][0])
                if bestWeight < regionWeight:
                    bestWeight = regionWeight
                    bestRegionIndex = i
        else:
            # Looping on all ROIs to calculate the weight of each one
            for i in range(len(roiRegions)):
                # Get the ROI as a 2D colored image (with 3 channels)
                regionImage = weightedEdges[roiRegions[i][0] : roiRegions[i][1], :]
                regionWeight = 0
                for k in range(regionImage.shape[0]):
                    prevEdge = 0
                    # Ignore the extreme left and right of the image
                    for j in range(20, regionImage.shape[1] - 20):
                        if regionImage[k][j] != 0:
                            if prevEdge == 0:
                                regionWeight += 1
                            else:
                                # As the distance between the current edge and previous edge increases
                                # the weight added for the current region decreases (negative exponential)
                                dist = np.abs(prevEdge - j) 
                                regionWeight += 1 / np.exp(dist)
                            prevEdge = j
                # Divide the region weight by the height of the ROI for fair comparison
                regionWeight /= (roiRegions[i][1] - roiRegions[i][0])
                # If the calculated weight is better than the best weight calculated, make it the best weight
                if bestWeight < regionWeight:
                    bestWeight = regionWeight
                    bestRegionIndex = i
        
        return roiRegions[bestRegionIndex]

    @staticmethod
    def extractLicense(image):

        # Apply some preprocessing to the region of interest in order to get the edges        
        image = image.astype('uint8')
        gray = cv2.bilateralFilter(image, 11, 17, 17)
        
        edged = sobel(gray)
        th = threshold_otsu(np.abs(edged))
        edged = edged > th
        edged = edged.astype('uint8')

        # Find contours based on Edges of ROI, 
        # The other two parameters are the contour heirarchy (external boundaries),
        # and contour approximation which removes the redundant points in this case
        contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Concerned only with the boundary points of the contour
        contours = contours[0] if len(contours) == 2 else contours[1]
        # Sort the greatest 50 given contours based on the contour area in descending order
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:50]

        # A list for holding candidates for being the license plate
        detectedPlates = []
        # Looping on each contour
        for contour in contours:
            # Given the contour (2D point set),
            # returns the rectangle [center(x, y), (width, height), angleOfRotation]
            rectangle = cv2.minAreaRect(contour)
            # Given the rectangle, returns a set of points ordered in clockwise direction
            box = np.int0(cv2.boxPoints(rectangle))

            # By considering two points on the same diagonal (any diagonal)
            _, start, _, end = box
            # Calculating the width and height given the points of the ends of the diagonal
            width = np.abs(end[0] - start[0])
            height = np.abs(end[1] - start[1])

            # Consider only the if the region has a width greater than 150 (By trial) and greater than the height
            if width > height and width > 150:
                xStart = min(start[0], end[0])
                xEnd = max(start[0], end[0])

                # Cropping the image on the x-axis only
                detectedLicensePlate = image[:, xStart : xEnd]

                # Collecting the points in two vertical edges (strating edge and ending edge)
                sortedBasedOnX = box[np.argsort(box[:, 0])]
                pointsOfStartingVerticalEdge = sortedBasedOnX[0 : 2]
                pointsOfEndingVerticalEdge = sortedBasedOnX[2 :]

                # Get the lower points of the two vertical edges
                lowerPointOfStart = pointsOfStartingVerticalEdge[np.argsort(pointsOfStartingVerticalEdge[:, 1])][0]
                lowerPointOfEnd = pointsOfEndingVerticalEdge[np.argsort(pointsOfEndingVerticalEdge[:, 1])][0]
            
                # Getting the angle of rotation of the horizontal edges using the slope of the lower edge -> (y2 - y1) / (x2 - x1)
                angleOfRotation = np.rad2deg(np.arctan2(lowerPointOfEnd[1] - lowerPointOfStart[1], lowerPointOfEnd[0] - lowerPointOfStart[0]))
                detectedLicensePlate = ndimage.rotate(detectedLicensePlate, angleOfRotation, cval = 255)
                # return detectedLicensePlate
                detectedPlates.append(detectedLicensePlate)
        # If there are no plates detected according to te previous algorithm
        # Take (hopelessly) the whole the middle third of the ROI and the whole ROI 
        if (len(detectedPlates) == 0):
            widthOfImage = gray.shape[1]
            detectedPlates.append(gray[:, widthOfImage // 4 : 3 * widthOfImage // 4])
            detectedPlates.append(gray)
        return detectedPlates

    @staticmethod
    def characterSegmentation(images):
        candidates = []
        for image in images:
            licensePlateRecognition = image.copy()

            # Image binarization using the OTSU which gets the optimal threshold based on the image histogram
            threshold = threshold_otsu(licensePlateRecognition)
            licensePlateRecognition[licensePlateRecognition <= threshold] = 0
            licensePlateRecognition[licensePlateRecognition > threshold] = 255
            contours, _ = cv2.findContours(licensePlateRecognition, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # Given the contour, get the value of x
            xFunction = lambda ctr: cv2.boundingRect(ctr)[0]
            # Sort the contours based on the value of x (from left to right)
            contours = sorted(contours, key = xFunction)

            # Contains the path (as raw string where backslash '\' is needed) to the tesseract program
            pytesseract.pytesseract.tesseract_cmd = r'D:\\Programs\\Tesseract-OCR\\tesseract.exe'
            # Setting the available letters and tessaract Configuration
            availableText = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            psmMode = 10
            # psm is the page segmentation mode, consider PSM 7 for LPR
            tesseractConfig = f"-c tessedit_char_whitelist={availableText} --psm {psmMode}"
            detectedLprText = ''
            characters = []
            for contour in contours:
            # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                if w * h < 800 and w * h > 200:

                    # Getting ROI
                    roi = licensePlateRecognition[y : y + h, x : x + w]
                    roi = np.pad(roi, 1, constant_values = 255)
                    roi = cv2.erode(roi, None)
                    text = pytesseract.image_to_string(roi, config = tesseractConfig, lang = "eng").split()
                    if len(text) == 1 and text[0] in availableText:
                        characters.append(roi)
                        detectedLprText += text[0]
            
            #concatenate segmented characters
            if (len(characters) == 0):
                continue
            minShape = sorted( [(np.sum(i.size), i.size ) for i in characters])[0]
            charactersComb = []
            for char in characters:
                charactersComb.append(cv2.resize(char,minShape, interpolation = cv2.INTER_CUBIC))

            segmentedChar = cv2.hconcat(charactersComb)
            psmMode = 7
            tesseractConfig = f"-c tessedit_char_whitelist={availableText} --psm {psmMode}"
            text = pytesseract.image_to_string(licensePlateRecognition, config = tesseractConfig, lang = "eng")
            print('Full Licence Photo', text)
            print('Segmented Character', detectedLprText)
            candidates.append([licensePlateRecognition, segmentedChar, detectedLprText])
        return candidates

    @staticmethod
    def detectLicense(image, grayImage):
        verticalEdges = LicenceDetection.detectVerticalEdges(image)
        weightedEdges = LicenceDetection.getWeightedEdges(verticalEdges)
        initialRoiRegion = LicenceDetection.initialRoiRegion(weightedEdges,image)
        bestRoi = LicenceDetection.getBestRegion(initialRoiRegion, weightedEdges, image)

        if bestRoi[0] < LicenceDetection.increase_number:
            bestRoi[0] = 0
        else:
            bestRoi[0] -= LicenceDetection.increase_number
        if image.shape[0] - bestRoi[1] < LicenceDetection.increase_number:
            bestRoi[1] = image.shape[0]
        else:
            bestRoi[1] += LicenceDetection.increase_number

        detectedLPR = LicenceDetection.extractLicense(grayImage[bestRoi[0]: bestRoi[1], :])

        # In case there are no candidates detected
        if len(detectedLPR) == 0:
            return None, None, None, None
        LicenseCandidates = LicenceDetection.characterSegmentation(detectedLPR)
        return detectedLPR[0], LicenseCandidates[0][0], LicenseCandidates[0][1], LicenseCandidates[0][2]