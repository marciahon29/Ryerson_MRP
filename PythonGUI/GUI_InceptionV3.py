from Tkinter import Tk, Label, Button
import tkFileDialog
from PIL import ImageTk, Image
import Tkinter as tk


from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np

file_path_string=""
model = InceptionV3(weights='imagenet', include_top=True)

class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.title("Alzheimer's Classification with Convolutional Neural Network")

        self.label = Label(master, text="Select the JPEG MRI brain image")
        self.label.pack()

        self.open_button = Button(master, text="Open File", command=self.open)
        self.open_button.pack()
        
        self.close_button = Button(master, text="Close Program", command=master.quit)
        self.close_button.pack()

        
    def open(self):
        for ele in root.winfo_children():
            ele.destroy()
            
        myFormats = [('JPEG / JFIF','*.jpg')]
        global file_path_string
        file_path_string = tkFileDialog.askopenfilename(filetypes=myFormats)
        im = Image.open(file_path_string)
        tkimage = ImageTk.PhotoImage(im)
        myvar = Label(root,image = tkimage)
        myvar.image = tkimage
        myvar.pack()

        global file_path_string
        img_path = file_path_string
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        global model
        preds = model.predict(x)
        
        global file_path_string
        myvar = Label(root, text=file_path_string + "\n" + str(decode_predictions(preds, top=3)[0]))
        myvar.pack()

        root.close_button = Button(root, text="Close Program", command=root.quit)
        root.close_button.pack()
        
root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()
