# Author: Angelo C. Guiam
# Exercise 02, CMSC 265, 2S 2020-2021

import cv2
import numpy
import os

TRAINING_IMG_DIR = 'raw_training/'
TOTAL_TRAINING_IMG = 66
TRAINING_FILE_EXT = '.jpeg'
TRAINING_EXPORT_DIR = 'data_training'

TEST_IMG_DIR = 'raw_test/'
TOTAL_TEST_IMG = 21
TEST_FILE_EXT = '.png'
TEST_EXPORT_DIR = 'data_test'

FILENAME_PREFIX = 'image-'
OUTPUT_PREFIX = 'output-'
IMG_DIM = 255

#remove the red markings in the raw dataset
def removeRedInk(image):
    white = [255, 255, 255]
    black = [0, 0, 0]

    (row, col, ch) = image.shape
    #traverse the whiole image
    for i in range(0, row):
        for j in range(0, col):
            bgr = image[i][j]
            #silence the pixel if it has high R value
            if(bgr[2] > 170):
                image[i][j] = white
    #returns a clean image
    return image


#crops and export section of an image
def exportImageSegment(image, x, y, w, h, filename, folder):
    #reduce bound values to crop out the rectangular bounding box from image
    x = x+5
    y = y+5
    w = w-10
    h = h-10

    #extract a section from original image and write result
    try:
        crop_image = image[y:y+h, x:x+w]
        resized_image = cv2.resize(crop_image, (IMG_DIM, IMG_DIM))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        #binarize image for a cleaner result
        ret, binary_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        # cv2.imwrite(filename, binary_image)
        cv2.imwrite(os.path.join(folder, filename), binary_image)
    except Exception as e:
        print(str(e))

    return


def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)


def generateEdges(image):
    #silence red markings in the image to improve edge detection
    clean = removeRedInk(image)
    #convert colored image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blur the image before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #get the edges using Canny edge detection algorithm
    return cv2.Canny(gray, 10, 100)


#detect rectangular shapes in the training data
def generateTrainingData():
    for i in range(1, 2):
        #load the image
        input_uri = TRAINING_IMG_DIR + FILENAME_PREFIX + f'{i:03}' + TRAINING_FILE_EXT
        image = cv2.imread(input_uri)
        #create edges using Canny edge detection algorithm
        edges = generateEdges(image)
        #find contours from the edges
        (cnts, heirarchy) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #create a base image for segmentation
        output = removeRedInk(image.copy())    
        #iterate all contours
        for j, cnt in enumerate(cnts):
            x,y,w,h = cv2.boundingRect(cnt)
            area = w*h

            filename = OUTPUT_PREFIX + FILENAME_PREFIX + f'{i:03}_{j:03}' + TRAINING_FILE_EXT
            #filter out bigger contours, assumed as the rectangle areas
            if w > 50 and h > 50 and y > 650 and y < 1100 and x > 70:
                print(f"filename={filename} x={x}, y={y}, w={w}, h={h}, area={area}")
                #some contours may contain
                chars_count = round(w / 65)
                #export images if the contour is a block of chars
                for char_idx in range(0, chars_count):
                    exportImageSegment(output.copy(), x + (char_idx*65), y, round(w/chars_count), h, filename, TRAINING_EXPORT_DIR)


#detect rectangular shapes in the training data
def generateTestData():
    for i in range(1, TOTAL_TEST_IMG+1):
        #load image
        input_uri = TEST_IMG_DIR + f'{i:03}' + TEST_FILE_EXT
        image = cv2.imread(input_uri)
        #generate edges using Canny edge detection algorithm
        edges = generateEdges(image.copy())
        #extract and sort contours from left to right
        (cnts, heirarchy) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (cnts, bounds) = sort_contours(cnts, "left-to-right")
        #create baseline image for segmentation
        output = image.copy()
        #will store the character's edges in rows 1 and 2 separately
        target_list_row_1 = []
        target_list_row_2 = []
        #x position marker
        last_x_row_1 = 0
        last_x_row_2 = 0

        for cnt_idx, cnt in enumerate(cnts):
            #unpack the bounds from each contour
            x,y,w,h = bounds[cnt_idx]

            #filter contour assumed to be rectangles in first row
            if y < 380 and w > 40 and h > 40:
                chars = round(w / 55)
                #come contours bound multiple characters, that needs to be segmented individually
                for char_index in range(0, chars):
                    if ((x + (char_index*55)) - last_x_row_1) > 10:
                        target_list_row_1.append((x + (char_index*55), y, round(w/chars), h))
                        last_x_row_1 = x + (char_index*55)

            #filter contour assumed to be rectangles in second row            
            elif y > 380 and w > 40 and h > 40:
                chars = round(w / 55)
                #come contours bound multiple characters, that needs to be segmented individually
                for char_index in range(0, chars):                        
                    if ((x + (char_index*55)) - last_x_row_2) > 10:
                        target_list_row_2.append((x + (char_index*55), y, round(w/chars), h))
                        last_x_row_2 = x + (char_index*55)
        #merge the sorted filtered contours    
        target_list = target_list_row_1 + target_list_row_2
        #iterate the extracted contours for segmentation
        for idx, item in enumerate(target_list):
            x,y,w,h = item
            area = w*h
            #automate sequential filename
            filename = f'{i:03}_{idx+1:03}' + TEST_FILE_EXT
            folder = f'{TEST_EXPORT_DIR}/{i:03}/test'
            print(f"filename={filename} x={x}, y={y}, w={w}, h={h}, area={area}")
            #image segmentation given the bounds of each contour
            exportImageSegment(image.copy(), x, y, w, h, filename, folder)


if not os.path.exists(TRAINING_EXPORT_DIR):
    print("Training folder created.")
    os.mkdir(TRAINING_EXPORT_DIR)

def createTestFolders():
    if not os.path.exists(TEST_EXPORT_DIR):
        os.mkdir(TEST_EXPORT_DIR)
        print("Test folder created.")

    for idx in range(0, TOTAL_TEST_IMG):
        folder_name = f'{TEST_EXPORT_DIR}/{idx+1:03}'
        sub_folder_name = f'{TEST_EXPORT_DIR}/{idx+1:03}/test'

        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        if not os.path.exists(sub_folder_name):
            os.mkdir(sub_folder_name)
            print(f"Test subfolder {sub_folder_name} created.")


createTestFolders()
generateTrainingData()
generateTestData()

