from  tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import optimizers
import cv2
#
import os
import shutil


model = load_model("W:\param's direc\models\\vgg_16_arch_model.h5")


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])
path="W:\\bucket_ludhiana_lane1\\2"
# path="W:\param's direc\\test"
mov_path="W:\param's direc\moved\incorrect_2"
dir = os.listdir( path )
i=1
for file in dir:
   new_img = path + '\\' + str(file)
   src=path + '\\' + str(file)
   img = load_img(new_img, target_size=(150, 150))
   print(type(img))
   print(file)
   # img.show()
   # # convert to array
   img = img_to_array(img)
   # # reshape into a single sample with 3 channels
   img = img.reshape(1, 150, 150, 3)
   #
   classes = model.predict_classes(img)
   c = int(classes)
   print(c)
   dst=mov_path+"\\"+str(file)
   i=i+1
   if c!=1:
       shutil.move(src, dst)


# new_img="W:\param's direc\\test\\1245.jpg"
