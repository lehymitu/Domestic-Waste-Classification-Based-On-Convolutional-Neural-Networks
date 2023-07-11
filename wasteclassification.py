import numpy as np
import cv2 as cv
import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from keras.losses import categorical_crossentropy
from PIL import Image, ImageTk
import tensorflow

model = tensorflow.keras.models.load_model('C:/Users/HP/Pictures/AIproject/wasteclassification.h5',compile=False)

classes = { 0: 'This is Cardboard',
            1: 'This is Glass',
            2: 'This is Metal',
            3: 'This is Paper',
            4: 'This is Plastic',
            5: 'This is Organic'}

# Create dictionary to store PhotoImage objects
image_dict = {}

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((200,200))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    
    predict_x=model.predict(image)
    predicted_class_index = np.argmax(predict_x, axis=1)[0]

    
    sign = classes[predicted_class_index]
    label_sign=tk.Label(root,text=sign,background='#b9dd7e',foreground='#2a773d', font=('intro rust base',30))
    label_sign.place(relx=0.2,rely=0.7)

def show_classify_button(file_path):
    # Check if PhotoImage object already exists in dictionary
    if file_path in image_dict:
        image_tk = image_dict[file_path]
    else:
        # Load image
        image = Image.open(file_path)
        image = image.resize((200,200))
        
        # Store image in PhotoImage object
        image_tk = ImageTk.PhotoImage(image)
        
        # Store reference to image in dictionary
        image_dict[file_path] = image_tk
    
    classify_b=tk.Button(root,text="Classify",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.config(background='#005237', foreground='#6dae49',font=('intro rust base',10))
    classify_b.place(relx=0.8,rely=0.6)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((root.winfo_width()/2.25),(root.winfo_height()/2.25)))
        
        # Check if PhotoImage object already exists in dictionary
        if file_path in image_dict:
            image_tk = image_dict[file_path]
        else:
            # Store image in PhotoImage object
            image_tk = ImageTk.PhotoImage(uploaded)
            
            # Store reference to image in dictionary
            image_dict[file_path] = image_tk
        
        label.config(image=image_tk)
        label.image=image_tk
        label.config(text='')
        show_classify_button(file_path)
    except:
        pass

def reset_interface():
    # Reset variables and processes
    # ...
    
    # Reload interface
    root.destroy()
    main()

def main():
    global root, label
    
    # Create main window
    root=tk.Tk()
    root.geometry('600x600')
    
    # Set background image
    background_image_path = "background.png"
    background_image = Image.open(background_image_path)
    background_image = background_image.resize((600,600))
    background_photo = ImageTk.PhotoImage(background_image)
    background_label = tk.Label(root, image=background_photo)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    
    # Add image label
    label=tk.Label(root,background='#b9dd7e', font=('notted regular',15,'bold'))
    label.pack(side=tk.BOTTOM,expand=True)
    
    # Add upload button
    upload=tk.Button(root,text="Upload an image",command=upload_image,padx=10,pady=5)
    upload.config(background='#6dae49', foreground='#005237',font=('intro rust base',10))
    upload.place(relx=1,rely=0.5,relwidth=0.25,anchor='ne')
    
    # Add classify and reset buttons
    reset = tk.Button(root, text="Try again", command=reset_interface, padx=10, pady=5)
    reset.config(background='#6dae49', foreground='#005237', font=('intro rust base', 10))
    reset.place(relx=0.5, rely=0.9, anchor=tk.CENTER)
    
    # Start main loop
    root.mainloop()

# Call main function to create interface
main()