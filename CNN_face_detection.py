import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

#libs needed for the input own images. 
import tkinter as tk
from tkinter.filedialog import askopenfilename

#load the model we trained
model = load_model('face_detection_cnn.h5')

#function to detect faces in webcam
def detectWebcam():
    #initialize a video capture object
    cap = cv.VideoCapture(0)

    while True:
        #we will real a single frame from video
        ret, frame = cap.read()

       
        #pre process the frame so it will match the input shape of the model in order to make predictions
        frame = cv.resize(frame, (200, 200))
        #store original dimensions of the frame in order to resize it back to original size to display it to user
        frame_dimentions = frame.shape[:2]
        #normalize the image
        # frame = frame / 255.0
        #add the batch dimension
        frame = np.expand_dims(frame,axis=0)


        #predict using the loaded model
        prediction = model.predict(frame)
        print(prediction)

        #Initialize the label according to the prediction made
        label = "Face Detected" if prediction > 0.8 else "No Face Detected"

        #resize and rescale it back in order to display the image
        frame = cv.resize(frame[0], frame_dimentions)
        # frame *= 255.0
        frame = frame.astype(np.uint8) 

        #display test saying detected or not
        cv.putText(frame, label, (10, 30), cv.LINE_4, 0.5, (0, 255, 0), 2)
        cv.imshow('face', frame)

        key = cv.waitKey(30) & 0xff
        if key == 27:
            break
    
    cap.release()
    cv.destroyAllWindows()



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

        

        img = cv.resize(img, (200, 200))
        img_dimentions = img.shape[:2]
        # img = img / 255.0
        img = np.expand_dims(img,axis=0)
        
        #convert to grey scale
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #detect faces
        prediction = model.predict(img)

        label = "Face Detected" if prediction > 0.8 else "No Face Detected"

        img = cv.resize(img[0], img_dimentions)
        # img *= 255.0

        img = img.astype(np.uint8) 

        cv.putText(img, label, (10, 30), cv.LINE_4, 0.5, (0, 255, 0), 2)

        #display image
        cv.imshow('img', img)

        cv.waitKey(0)

# selectImage()



#main function
print("Detect Faces with CNN")
print("How do you want to detect faces ?")
print("Webcam - 1\nSelect Own Image - 2\nQuit - 99\n")
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
        elif(user_input == 99):
            #quits the program
            break
        else:
            print('Invalid input')
    except ValueError:
        print('Invalid input')