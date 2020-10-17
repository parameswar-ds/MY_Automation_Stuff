import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import glob,cv2
direc=["train"]
fol=["1.Empty category"]
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
c=900000
# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')
for d in range(0,len(direc)):
    for f in range(0,len(fol)):
        files=glob.glob(f"D:\zzzz_params\\techable_data_tests\\new_automated_folders\\automated_by_model\\{direc[d]}\\{fol[f]}\\*.jpg")
        # print(len(files))
        for i in range(0,len(files)):

            # Create the array of the right shape to feed into the keras model
            # The 'length' or number of images you can put into the array is
            # determined by the first position in the shape tuple, in this case 1.
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            # Replace this with the path to your image
            image = Image.open(files[i])

            #resize the image to a 224x224 with the same strategy as in TM2:
            #resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)

            #turn the image into a numpy array
            image_array = np.asarray(image)

            # display the resized image
            # image.show()

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

            # Load the image into the array
            data[0] = normalized_image_array

            # run the inference
            prediction = model.predict(data)
            print(c)

            print(int(np.argmax(prediction)))
            # print(prediction[0][int(np.argmax(prediction))])
            if int(np.argmax(prediction))==0:
                img_s=cv2.imread(files[i])
                cv2.imwrite(f"D:\zzzz_params\\techable_data_tests\\new_automated_folders\\automated_by_1_to_itself\\empty_1\\{str(c)}.jpg",img_s)
                c=c+1
            else:
                img_s=cv2.imread(files[i])
                cv2.imwrite(f"D:\zzzz_params\\techable_data_tests\\new_automated_folders\\automated_by_1_to_itself\\empty_2\\{str(c)}.jpg",img_s)
                c=c+1

