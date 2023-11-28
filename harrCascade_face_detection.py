#--import opencv for face detection
import cv2 as cv
import numpy as np

from os import listdir
from os.path import isfile, join

import tkinter as tk
from tkinter.filedialog import askopenfilename

#load the pretrained classifier that i downloaded from the opencv git repo
front_face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

#creating a Tkinter window to display
root = tk.Tk()
#hide the main Tkinter window
root.withdraw()


#manually selecting an image from your computer
def selectImage():
        #select the image
        image_path = askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*png;*jpeg")])

        #check if the image is selected properly
        if not image_path:
            print("No file selected")
            exit()
        
        #load image to variable
        img = cv.imread(f'{image_path}')
        
        #convert to grey scale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #detect faces
        faces = front_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        #drawing the rectangle on the detected faces
        img = drawRectangle(img, faces)

        #display image
        cv.imshow('img', img)

        cv.waitKey(0)
        




#function to detect faces from webcam
def detectWebcam():

    #initialize video capture object. 0 means default camera is used.
    cap = cv.VideoCapture(0) 

    while True:
        #_ is a flag to see if the image was read or not
        #img is the frame
        _, img = cap.read()

        #convert the frame to grayscale
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #use the pre-trained classifier and identify faces using grayscale images
        faces = front_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1,minNeighbors=4)

        #call the function which draws the rectangles on detected faces
        img = drawRectangle(img, faces)

        #display the image with detected face
        cv.imshow('img', img)

        #when escape key is pressed, the loop will break
        key = cv.waitKey(30) & 0xff
        if key == 27:
            break
    
    #video capture object is released
    cap.release()


#function to draw rectangles on detected faces
def drawRectangle(img, faces):
    #iterate through a loop. enumerate all faces in order to name faces separately when multiple faces are detected
    for i, (x, y, w, h) in enumerate(faces):
        #(x, y) - left top corner of rectangle and (x+w, y+h) is the right bottom corner of rectangle
        cv.rectangle(img, (x, y), (x+w, y+h), (0,0,255),2)
        text = f'Person {i+1}'
        cv.putText(img, text, (x, y - 10), cv.LINE_4, 0.5, (0, 0, 255), 2)
    return img



#function to read images from the file
def readTestImages():
    #load the images from test_images file
    file_path = 'test_images/'
    #get all the file paths to th images in a list
    files = [file for file in listdir(file_path) if isfile(join(file_path, file))]
    #creating a numpy array
    images = np.empty(len(files), dtype=object)
    #initialize a list for gray images
    gray_images = []
    #read all the images and convert to gray scale and append inside this for loop
    for i in range(0, len(files)):
        images[i] = cv.imread(join(file_path, files[i]))
        gray_image = cv.cvtColor(images[i], cv.COLOR_BGR2GRAY)
        gray_images.append(gray_image)
    return images, gray_images


#function to detect faces from test images
def detectImage():
    #read all images in image database
    pic, gray_pic = readTestImages()
    #iterate through all the images and detect faces
    for i in range(len(pic)):
        cv.imshow('img', pic[i])
        faces = front_face_cascade.detectMultiScale(gray_pic[i], scaleFactor=1.3, minNeighbors=5)
        #iterate through all the faces
        for j, (x, y, w, h) in enumerate(faces):
            cv.rectangle(pic[i], (x, y), (x+w, y+h), (255,0,0),2)
            text = f'Person {j+1}'
            cv.putText(pic[i], text, (x, y - 10), cv.LINE_4, 0.5, (0, 0, 255), 2)
        # drawRectangle(pic,faces)

        #display the image
        cv.imshow('img', pic[i])
        cv.waitKey()



#main function
print("How do you want to detect faces ?")
print("Webcam - 1\nSelect Own Image - 2\nUse all images in database - 3\nQuit - 99\n")
while input !=99:
    try:
        user_input = int(input('Enter a number only: '))
        if(user_input == 1):
            #detect images from webcam
            detectWebcam()
            break
        elif(user_input == 2):
            #detect images inputted by user
            selectImage()
            break
        elif(user_input == 3):
            #detect all images from database/file
            detectImage()
            break
        elif(user_input == 99):
            #quits the program
            break
        else:
            print('Invalid input')
    except ValueError:
        print('Invalid input')   