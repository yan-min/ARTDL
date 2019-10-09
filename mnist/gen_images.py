import os
import time
import csv

import numpy as np
import random
from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave
from skimage import exposure, img_as_float
from PIL import Image

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3

import cv2
import scipy.misc

# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(_, _), (x_test, _) = mnist.load_data()

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
model1 = Model1(input_tensor=input_tensor)
model2 = Model2(input_tensor=input_tensor)
model3 = Model3(input_tensor=input_tensor)

def image_translation(img, params):
    rows,cols,ch = img.shape

    M = np.float32([[1,0,params[0]],[0,1,params[1]]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def image_brightness(img, params):
    new_img = exposure.adjust_gamma(img,params)
  
    return new_img

def image_blur(img, params):
    blur = []
    if params == 1:
        blur = cv2.blur(img,(3,3))
    if params == 2:
        blur = cv2.blur(img,(4,4))
    if params == 3:
        blur = cv2.blur(img,(5,5))
    if params == 4:
        blur = cv2.GaussianBlur(img,(3,3),0)
#     if params == 5:
#         blur = cv2.GaussianBlur(img,(4,4),0)
    if params == 5:
        blur = cv2.GaussianBlur(img,(5,5),0)
    if params == 6:
        blur = cv2.medianBlur(img,3)
#     if params == 8:
#         blur = cv2.medianBlur(img,4)
    if params == 7:
        blur = cv2.medianBlur(img,5)
    if params == 8:
        blur = cv2.bilateralFilter(img,5,50,50)
    return blur        
def image_motionBlur(image, degree, angle):
    image = np.array(image)
 
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
 
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
 
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred
if __name__ == "__main__":
    now_time = time.strftime("%Y%m%d-%H%M%S",time.localtime())
    with open('./mnist_model_gen'+now_time+'_images.csv', 'wb',0) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['index', 'tranformation','param_value','y_before','y_after','is_correct'])
        
        # translation
        for p in xrange(1, 10):
            params = [p*0.25, p*0.25]
            for i in range(0,10000):
                csvrecord = []
                source_image = np.expand_dims(x_test[i], axis=0)
                predictions1 = np.argmax(model3.predict(source_image)[0])
                gen_image = image_translation(source_image[0],params)
                predictions2 = np.argmax(model3.predict(gen_image.reshape(1,28,28,1))[0])
                if predictions1 == predictions2:
                    folder = "./gen_images/correct/"+str(predictions1)+"/"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    name = "translation_"+str(p)+"_"+str(i)+".jpg"
                    image_path = os.path.join(folder, name)
                    scipy.misc.imsave(image_path, gen_image.reshape(28,28))
                    is_correct = 1
                else:
                    folder = "./gen_images/error/"+str(predictions1)+"/"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    name = "translation_"+str(p)+"_"+str(i)+"_"+str(predictions2)+".jpg"
                    image_path = os.path.join(folder, name)
                    scipy.misc.imsave(image_path, gen_image.reshape(28,28))
                    is_correct = 0
                csvrecord.append(i)
                csvrecord.append('translation')
                csvrecord.append(p)

                csvrecord.append(predictions1)
                csvrecord.append(predictions2)
                csvrecord.append(is_correct)
                print(csvrecord)
                writer.writerow(csvrecord)
        print("translation done")


        # moveBlur
        for p in xrange(1, 7):
            params = p
            for i in range(0,10000):
                csvrecord = []
                source_image = np.expand_dims(x_test[i], axis=0)
                predictions1 = np.argmax(model3.predict(source_image)[0])
                angle = 45
                gen_image = image_motionBlur(source_image[0],params,angle)
                predictions2 = np.argmax(model3.predict(gen_image.reshape(1,28,28,1))[0])
                if predictions1 == predictions2:
                    folder = "./gen_images/correct/"+str(predictions1)+"/"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    name = "moveBlur_"+str(p)+"_"+str(i)+".jpg"
                    image_path = os.path.join(folder, name)
                    scipy.misc.imsave(image_path, gen_image.reshape(28,28))
                    is_correct = 1
                else:
                    folder = "./gen_images/error/"+str(predictions1)+"/"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    name = "moveBlur_"+str(p)+"_"+str(i)+"_"+str(predictions2)+".jpg"
                    image_path = os.path.join(folder, name)
                    scipy.misc.imsave(image_path, gen_image.reshape(28,28))
                    is_correct = 0
                csvrecord.append(i)
                csvrecord.append('moveBlur')
                csvrecord.append(params)

                csvrecord.append(predictions1)
                csvrecord.append(predictions2)
                csvrecord.append(is_correct)
                print(csvrecord)
                writer.writerow(csvrecord)
        print("moveBlur done")



        # Brightness
        for p in xrange(1, 11):
            params = p
            for i in range(0,10000):
                csvrecord = []
                source_image = np.expand_dims(x_test[i], axis=0)
                scipy.misc.imsave("./source_mnist.jpg", source_image.reshape(28,28))
                predictions1 = np.argmax(model3.predict(source_image)[0])
                gen_image = image_brightness(source_image[0],params)
                scipy.misc.imsave("./gen_mnist.jpg", gen_image.reshape(28,28))
                predictions2 = np.argmax(model3.predict(gen_image.reshape(1,28,28,1))[0])
                if predictions1 == predictions2:
                    folder = "./gen_images/correct/"+str(predictions1)+"/"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    name = "brightness_"+str(p)+"_"+str(i)+".jpg"
                    image_path = os.path.join(folder, name)
                    scipy.misc.imsave(image_path, gen_image.reshape(28,28))
                    is_correct = 1
                else:
                    folder = "./gen_images/error/"+str(predictions1)+"/"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    name = "brightness_"+str(p)+"_"+str(i)+"_"+str(predictions2)+".jpg"
                    image_path = os.path.join(folder, name)
                    scipy.misc.imsave(image_path, gen_image.reshape(28,28))
                    is_correct = 0
                csvrecord.append(i)
                csvrecord.append('brightness')
                csvrecord.append(params)

                csvrecord.append(predictions1)
                csvrecord.append(predictions2)
                csvrecord.append(is_correct)
                print(csvrecord)
                writer.writerow(csvrecord)
        print("brightness done")

        # blur
        for p in xrange(1, 9):
            params = p
            for i in range(0,10000):
                csvrecord = []
                source_image = np.expand_dims(x_test[i], axis=0)
                predictions1 = np.argmax(model3.predict(source_image)[0])
                gen_image = image_blur(source_image[0],params)
        #         scipy.misc.imsave("./mnist.jpg", gen_image.reshape(gen_image.shape[0],gen_image.shape[0]))
                predictions2 = np.argmax(model3.predict(gen_image.reshape(1,28,28,1))[0])
                if predictions1 == predictions2:
                    folder = "./gen_images3/correct/"+str(predictions1)+"/"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    name = "blur_"+str(p)+"_"+str(i)+".jpg"
                    image_path = os.path.join(folder, name)
                    scipy.misc.imsave(image_path, gen_image.reshape(28,28))
                    is_correct = 1
                else:
                    folder = "./gen_images3/error/"+str(predictions1)+"/"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    name = "blur_"+str(p)+"_"+str(i)+"_"+str(predictions2)+".jpg"
                    image_path = os.path.join(folder, name)
                    scipy.misc.imsave(image_path, gen_image.reshape(28,28))
                    is_correct = 0
                csvrecord.append(i)
                csvrecord.append('blur')
                csvrecord.append(params)

                csvrecord.append(predictions1)
                csvrecord.append(predictions2)
                csvrecord.append(is_correct)
                print(csvrecord)
                writer.writerow(csvrecord)
        print("blur done")
        
        
        
        
        
        
        
        
        
        
        