# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 21:45:47 2023

@author: SUWARNA
"""
#E:\Data Science Study materials and BOOKS\Deep learning training\We\Train
from tkinter import *
from functools import partial
from tkinter.ttk import *
from PIL import ImageTk
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pathlib
from IPython.display import Image
from tensorflow import keras
import PIL.Image

def setDir(directory):
    direct = r'' + directory.get()
    dir_ = pathlib.Path(direct)
    return dir_

	
def ml_tkinter_gui():
    data_dir=setDir(username)
    img_height,img_width=180,180
    batch_size=32
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

    #TRANSFER LEARNING
    resnet_model = Sequential()

    pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                       input_shape=(180,180,3),
                       pooling='avg',classes=2,
                       weights='imagenet')
    for layer in pretrained_model.layers:
            layer.trainable=False

    resnet_model.add(pretrained_model)

    resnet_model.add(Flatten())
    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(Dense(2, activation='softmax'))

    resnet_model.summary()

    #Checking GPU Availabilty
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    #Training Resnet50 Model
    resnet_model.compile(optimizer=Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    history = resnet_model.fit(train_ds, validation_data=val_ds, epochs=15)
    model=resnet_model.save("WEresnet50.h5")
    print("Saved model to disk")
    return model
    
    

########################################################################################################

#window
tkWindow = Tk()  
tkWindow.geometry('1200x700')  
tkWindow.title('HORIZON')

##tkWindow['background']='#856ff8'

Label(tkWindow, text = "HORIZON",
         font=('calibri',40,'bold')).grid(column=1, row=0, padx=10, pady=25)

style = Style()
 
style.configure('TButton', font =
               ('calibri', 20, 'bold'),
                    borderwidth = '4')
 
# Changes will be reflected
# by the movement of mouse.
style.map('TButton', foreground = [('active', '!disabled', 'green')],
                     background = [('active', 'black')])

#test label 
usernameLabel = Label(tkWindow, text="Enter Training Data Directory",font=('arial',12,'bold')).grid(row=1, column=0,padx=10, pady=25)
username = StringVar()
usernameEntry = Entry(tkWindow, textvariable=username,width=140).grid(row=1, column=1)  

loginButton = Button(tkWindow, text="Train",command=ml_tkinter_gui ).grid(row=2, column=1)
#ml_tkinter_gui  command=myClick
#training label 
passwordLabel = Label(tkWindow,text="Enter Testing File Directory",font=('arial',12,'bold')).grid(row=4, column=0,padx=10, pady=25)  
password = StringVar()
passwordEntry = Entry(tkWindow, textvariable=password, width=140).grid(row=4, column=1)  



my_label = Label()
my_label.grid(row=10, column=1) 

#read the image, creating an object
Class1 = Label(text="Class-1",font=('arial',12,'bold'))  
Class1.place(relx=0.1, rely=0.6, anchor='center')
C1 = StringVar()
Ce1 = Entry(tkWindow, textvariable=C1, width=20)  
Ce1.place(relx=0.3, rely=0.6, anchor='center')

Class2 = Label(text="Class-2",font=('arial',12,'bold'))  
Class2.place(relx=0.1, rely=0.8, anchor='center')
C2 = StringVar()
Ce2 = Entry(tkWindow, textvariable=C2, width=20)
Ce2.place(relx=0.3, rely=0.8, anchor='center')

def myClick(link):    
    img=PIL.Image.open(link)
    resize_image = img.resize((250,250))
    img = ImageTk.PhotoImage(resize_image)
    my_label.configure(image=img)
    my_label.image = img

def predict_img():
    img_path = r'' + password.get()
    showIm=myClick(img_path)
    image = tf.keras.preprocessing.image.load_img(img_path, target_size=(180, 180))
    image_p = tf.keras.preprocessing.image.img_to_array(image)
    image_p = image_p.reshape((1, image_p.shape[0], image_p.shape[1], image_p.shape[2]))
    image_p = tf.keras.applications.resnet.preprocess_input(image_p)
    model=keras.models.load_model("WEresnet50.h5")
    pred = model.predict(image_p)
    pred_string = C1.get() if pred[0][0] > 0.5 else C2.get()
    result_label.config(text=f'Prediction is: {pred_string}\nConfidence is: {max(pred[0])}')
    return Image(filename=img_path)


result_label = Label(text="",font=('arial',15,'bold'))  
result_label.place(relx=0.8, rely=0.8, anchor='center') 
loginButton = Button(tkWindow, text="Test", command=predict_img).grid(row=6, column=1,padx=10, pady=25)

#C:\Users\SUWARNA\Pictures\Camera Roll\Me.jpg
tkWindow.mainloop()