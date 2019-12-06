# modules used
import cv2
import numpy as np
import imutils
import argparse
from imutils import paths
import csv
import os
import matplotlib.pyplot as plt
from imutils import contours
from skimage import measure
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

#display Windows and processed image results
def storeImage(rimg, image):

    path = "/home/lucidgravity/Documents/PyScript"
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.savefig(os.path.join(path, rimg), dpi=1000)
    plt.show()

#analytical plots for the processed images
def plotImage(label, area, labl, leng_imgs, img_indx, rplt):

    path = "/home/lucidgravity/Documents/PyScript"
    fig, ax = plt.subplots()
    ax.plot(label, area, linewidth=1, marker='.', markersize=1)
    plt.bar(label,area, alpha=0.2)
    plt.xlabel('Droplet labels', fontsize=7)
    plt.ylabel('Droplet areas per frame (in pixels)', fontsize=7)

    ax.set_ylim(0,400)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.yticks(fontsize=5)
    plt.xticks(label, labl, rotation='vertical', fontsize=3)
    plt.savefig(os.path.join(path, rplt), dpi=1000)
#    if leng_imgs == (img_indx + 1):
    plt.show()


#argument parsing for command line invocation
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required = True, help = "Path to the input image directory")
args = vars(ap.parse_args())

#retrieving image paths
imgPaths = sorted(list(paths.list_images(args["images"])))
images = []
filenames = []

#extracting image list from retrieved image paths from directory
for imgPath in imgPaths:
    image = cv2.imread(imgPath)
    images.append(image)
    filenames.append(os.path.basename(imgPath)[:8])

#reading the image, resizing and optimizing for further operations
imfils = list(zip(images, filenames))
len_imgs = len(images)

for i, imfil in list(enumerate(imfils)):

    image = imfil[0]
    filename = imfil[1]
    img_idx = i

#cropping the original image to 20% width
    timg = imutils.resize(image, 1000)
    H, W = timg.shape[:2]
    img = timg[0:H, int(W*.20):W]

#processing every image for detection

    image_cpy = img.copy()
    opr = imutils.resize(image_cpy, 1000)

#Section for dish detection  ##former approach for droplet coordinate reference
#######################################################
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #cir = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1.85, 300, param1= 200, param2 = 40, minRadius=270, maxRadius=273)
    #if cir is not None:
    #    cir = np.round(cir[0,:]).astype('int')
    #    for (x, y, r) in cir:
    #        cv2.circle(opr, (x, y), r, (255, 255, 255), 0)
#######################################################

#converting to HSV to apply masking and removing unwanted uv light reflections

    hsv_img = cv2.cvtColor(opr, cv2.COLOR_BGR2HSV)
    green_low = np.array([25, 52, 72] )
    green_high = np.array([102, 255, 255])
    curr_mask = cv2.inRange(hsv_img, green_low, green_high)
    bitimg = cv2.bitwise_and(hsv_img, hsv_img, mask = curr_mask)


#converting the HSV image to Gray and thresholding in order to be able to apply contouring

    RGB_again = cv2.cvtColor(bitimg, cv2.COLOR_HSV2RGB)
    hgray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)
    oprth = cv2.threshold(hgray, 1, 255, cv2.THRESH_BINARY)[1]

#generating labels in the thresholded image using skimage module

    labels = measure.label(oprth, connectivity=2, background=0)
    mask = np.zeros(oprth.shape, dtype="uint8")

    for label in np.unique(labels):

        if label == 0:
            continue

#masking of thresholded image for significant detections

        labelMask = np.zeros(oprth.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        if numPixels > 0:
            mask = cv2.add(mask, labelMask)

#Contouring using the mask

    cnts =  cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]

#lists used for storing detected contour details

    drop_area = []  #Area
    cont_coord = [] #Coordinates
    label_cnt = [] #Contour labels
    label_txt = [] #Label texts

#marking and formatting of detected contours

    for (i,c) in enumerate(cnts):

        (x, y, w, h) = cv2.boundingRect(c)

        cv2.putText(opr, "#{}".format(i + 1), (x - 25, y - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.22, (255, 255, 0), 1)

#Section for determining contour areas and coordinates in terms of pixels
        M = cv2.moments(c)
        ar = cv2.contourArea(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX,cY = 0,0

        lbl = "#" + str(i+1)
        drop_area.append(ar)
        cont_coord.append((cX,cY))
        label_cnt.append(i+1)
        label_txt.append(lbl)

#Outlining the detected contours

    cv2.drawContours(opr, cnts, -1, (0, 0, 255), 1)

    number_of_droplets = len(cnts)
    text = "Droplet Count: " + str(len(cnts))
    cv2.putText(opr, text, (350, 850), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255), 1)

#writing results to the directory

#storing the processed image using the definition

    res_img = filename + '_PRCSD.jpg'
    storeImage(res_img, opr)

#plotting the results for the processed image using the definition

    res_plt = filename + '_Plotted.jpg'
    plotImage(label_cnt, drop_area, label_txt, len_imgs, img_idx, res_plt)

#writing the processed image details into CSV

    fname = filename + '.csv'

    with open(fname, 'w', newline = '') as f:

        thewriter = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar=' ')

        thewriter.writerow(['Droplet_Area','Droplet_Coordinates'])
        for dp_area, coord in zip(drop_area, cont_coord):
            thewriter.writerow([dp_area,coord])
