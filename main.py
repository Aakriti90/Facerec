import cv2
import os
from cv2 import COLOR_BGR2GRAY

files = os.listdir("positive-image")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # defining classifier #method ->cascade classifier
# Read the input image

for i in files:

    img = cv2.imread("./positive-image/" + i)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting image to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # method ->detectMultiscale to scale the size of image
    # gray->gray-scale image, 1.1->scaleFactor(how much image size is reduced at each image scale?),4->MInneighbour(how many neighbour each candidate rectangle should have to retain it)
    # faces variable will be the vector of rectangle where each reactangle contain that detected object
    if len(faces) == 0:
        print("no faces found in ", i)
    else:
        print("faces found in ", i)

    # iterate over this faces object ->drawing rectange
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)  # method->rectange parameter:img,(x,y) from faces vector , (x+w) (y+h),color , thickness
    # # Display the output
    cv2.imshow("img", img)  # display an image in windows
    cv2.waitKey(1000)  # after 10ms it will destroy the showing windows
