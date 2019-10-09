import glob
import argparse
import numpy as np
import time
from collections import deque
from keras.models import load_model
# from keras.models import Model as Kmodel
from keras.preprocessing.image import load_img, img_to_array
from skimage.exposure import rescale_intensity
from scipy import misc
from scipy.misc import imread, imresize, imsave
import sys
import os
import csv
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imshow
from scipy.ndimage.interpolation import map_coordinates

reload(sys)  
sys.setdefaultencoding('ISO-8859-1')

class Model(object):
    def __init__(self,
                 model_path,
                 X_train_mean_path):

        self.model = load_model(model_path)
        self.model.compile(optimizer="adam", loss="mse")
        self.X_mean = np.load(X_train_mean_path)
        self.mean_angle = np.array([-0.004179079])
        print (self.mean_angle)
        self.img0 = None
        self.state = deque(maxlen=2)

    def predict(self, img_path):
        #img_path = 'test.jpg'
        #misc.imsave(img_path, img)
        img1 = load_img(img_path, grayscale=True, target_size=(192, 256))
        img1 = img_to_array(img1)

        if self.img0 is None:
            self.img0 = img1
            return self.mean_angle

        elif len(self.state) < 1:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8) # to replicate initial model
            self.state.append(img)
            self.img0 = img1

            return self.mean_angle

        else:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8) # to replicate initial model
            self.state.append(img)
            self.img0 = img1

            X = np.concatenate(self.state, axis=-1)
            X = X[:, :, ::-1]
            X = np.expand_dims(X, axis=0)
            X = X.astype('float32')
            X -= self.X_mean
            X /= 255.0
            return self.model.predict(X)[0]

def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()
        
def image_translation(img, params):
    rows,cols,ch = img.shape

    M = np.float32([[1,0,params[0]],[0,1,params[1]]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def image_scale(img, params):

    res = cv2.resize(img,None,fx=params[0], fy=params[1], interpolation = cv2.INTER_CUBIC)
    return res

def image_shear(img, params):
    rows,cols,ch = img.shape
    factor = params*(-1.0)
    M = np.float32([[1,factor,0],[0,1,0]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def image_rotation(img, params):
    rows,cols,ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),params,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def image_contrast(img, params):
    alpha = params
    new_img = cv2.multiply(img, np.array([alpha]))                    # mul_img = img*alpha
    #new_img = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta
  
    return new_img

def image_brightness(img, params):
    beta = params
    new_img = cv2.add(img, beta)                                  # new_img = img*alpha + beta
  
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
    if params == 6:
        blur = cv2.GaussianBlur(img,(5,5),0)
    if params == 7:
        blur = cv2.medianBlur(img,3)
#     if params == 8:
#         blur = cv2.medianBlur(img,4)
    if params == 9:
        blur = cv2.medianBlur(img,5)
    if params == 10:
        blur = cv2.bilateralFilter(img,9,75,75)
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

def fog(x, severity=5):
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    x = x + c[0] * plasma_fractal(wibbledecay=c[1])[:256, :256]
    x = np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255
    return x.astype(np.float32)


def rambo_testgen_coverage(dataset_path):
    seed_inputs1 = os.path.join(dataset_path, "hmb3/")
    seed_labels1 = os.path.join(dataset_path, "hmb3/hmb3_steering.csv")
    seed_inputs2 = os.path.join(dataset_path, "Ch2_001/center/")
    seed_labels2 = os.path.join(dataset_path, "Ch2_001/CH2_final_evaluation.csv")

    
    model = Model("./final_model.hdf5", "./X_train_mean.npy")
    model_after = Model("./final_model.hdf5", "./X_train_mean.npy")
    filelist1 = []
    for file in sorted(os.listdir(seed_inputs1)):
        if file.endswith(".jpg"):
            filelist1.append(file)

    filelist2 = []
    for file in sorted(os.listdir(seed_inputs2)):
        if file.endswith(".jpg"):
            filelist2.append(file)
    
    with open(seed_labels1, 'rb') as csvfile1:
        label1 = list(csv.reader(csvfile1, delimiter=',', quotechar='|'))
    label1 = label1[1:]

    with open(seed_labels2, 'rb') as csvfile2:
        label2 = list(csv.reader(csvfile2, delimiter=',', quotechar='|'))
    label2 = label2[1:]

    

    now_time = time.strftime("%Y%m%d-%H%M%S",time.localtime())
    error_count = 0
    with open('result/rambo_new'+now_time+'_images.csv', 'wb',0) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['index', 'image', 'tranformation','param_value','y_before','y_after'])

        #move_blur   
        for p in xrange(1, 13):
            params = p
            for i in range(0,len(filelist2)):
                csvrecord = []
                seed_image_path = os.path.join(seed_inputs2, filelist2[i])
                seed_image = imread(os.path.join(seed_inputs2, filelist2[i]))
                angle = 45
                gen_image = image_motionBlur(seed_image,params,angle)
                name = 'motionBlur'+'_'+str(p)+'_'+str(filelist2[i])
                folder = "./new0109/motionBlur/"
                if not os.path.exists(folder):
                    os.makedirs(folder)
                after_image_path = os.path.join(folder, name)
                cv2.imwrite(after_image_path, gen_image)
                
                
                result_before = model.predict(seed_image_path)[0]
                result_after = model_after.predict(after_image_path)[0]

                csvrecord.append(i)
                csvrecord.append(name)
                csvrecord.append('motionBlur')
                csvrecord.append(params)

                csvrecord.append(result_before)
                csvrecord.append(result_after)
                print(csvrecord)
                
                writer.writerow(csvrecord)

        print("motionBlur done")    
        #Translation   
        for p in xrange(1, 11):
            params = [p*10, p*10]
            for i in range(0,len(filelist2)):
                csvrecord = []
                seed_image_path = os.path.join(seed_inputs2, filelist2[i])
                seed_image = imread(os.path.join(seed_inputs2, filelist2[i]))

                gen_image = image_translation(seed_image,params)
                name = 'translation'+'_'+str(p)+'_'+str(filelist2[i])
                folder = "./new0109/translation/"
                if not os.path.exists(folder):
                    os.makedirs(folder)
                after_image_path = os.path.join(folder, name)
                cv2.imwrite(after_image_path, gen_image)
                
                
                result_before = model.predict(seed_image_path)[0]
                result_after = model_after.predict(after_image_path)[0]

                csvrecord.append(i)
                csvrecord.append(name)
                csvrecord.append('translation')
                csvrecord.append(params)

                csvrecord.append(result_before)
                csvrecord.append(result_after)
                print(csvrecord)
                
                writer.writerow(csvrecord)

        print("translation done")
        
        #Scale
        for p in xrange(1, 11):
            params = [p*0.5+1, p*0.5+1]            

            for i in range(0,len(filelist2)):
                csvrecord = []
                seed_image_path = os.path.join(seed_inputs2, filelist2[i])
                seed_image = imread(os.path.join(seed_inputs2, filelist2[i]))

                gen_image = image_scale(seed_image,params)
                name = 'scale'+'_'+str(p)+'_'+str(filelist2[i])
                folder = "./new0109/scale/"
                if not os.path.exists(folder):
                    os.makedirs(folder)
                after_image_path = os.path.join(folder, name)
                cv2.imwrite(after_image_path, gen_image)
                
                
                result_before = model.predict(seed_image_path)[0]
                result_after = model_after.predict(after_image_path)[0]

                csvrecord.append(i)
                csvrecord.append(name)
                csvrecord.append('scale')
                csvrecord.append(params)

                csvrecord.append(result_before)
                csvrecord.append(result_after)
                print(csvrecord)
                
                writer.writerow(csvrecord)

        print("scale done")
      
        #Brightness
        input_images = xrange(1, 3001)
        for p in xrange(1, 11):
            params = p * 10 
            for i in range(0,len(filelist2)):
                csvrecord = []
                seed_image_path = os.path.join(seed_inputs2, filelist2[i])
                seed_image = imread(os.path.join(seed_inputs2, filelist2[i]))

                gen_image = image_brightness(seed_image,params)
                name = 'brightness'+'_'+str(p)+'_'+str(filelist2[i])
                folder = "./new0109/brightness/"
                if not os.path.exists(folder):
                    os.makedirs(folder)
                after_image_path = os.path.join(folder, name)
                cv2.imwrite(after_image_path, gen_image)
                
                
                result_before = model.predict(seed_image_path)[0]
                result_after = model_after.predict(after_image_path)[0]           


                csvrecord.append(i)
                csvrecord.append(name)
                csvrecord.append('brightness')
                csvrecord.append(params)


                csvrecord.append(result_before)
                csvrecord.append(result_after)
                print(csvrecord)
                writer.writerow(csvrecord)
        print("brightness done")
        
        
#         blur
        input_images = xrange(1, 3001)
        for p in xrange(1, 11):
            params = p
            if p != 5 and p != 8:
                for i in range(0,len(filelist2)):
                    csvrecord = []
                    seed_image_path = os.path.join(seed_inputs2, filelist2[i])
                    seed_image = imread(os.path.join(seed_inputs2, filelist2[i]))

                    gen_image = image_blur(seed_image,params)
                    name = 'blur'+'_'+str(p)+'_'+str(filelist2[i])
                    folder = "./new0109/blur/"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    after_image_path = os.path.join(folder, name)
                    cv2.imwrite(after_image_path, gen_image)

                    result_before = model.predict(seed_image_path)[0]
                    result_after = model_after.predict(after_image_path)[0]

                    param_name = ""
                    if params == 1:
                        param_name = "averaging:3:3"
                    if params == 2:
                        param_name = "averaging:4:4"
                    if params == 3:
                        param_name = "averaging:5:5"
                    if params == 4:
                        param_name = "GaussianBlur:3:3"
                    if params == 5:
                        param_name = "GaussianBlur:4:4"
                    if params == 6:
                        param_name = "GaussianBlur:5:5"
                    if params == 7:
                        param_name = "medianBlur:3"
                    if params == 8:
                        param_name = "medianBlur:4"
                    if params == 9:
                        param_name = "medianBlur:5"
                    if params == 10:
                        param_name = "bilateralFilter:9:75:75"


                    csvrecord.append(i)
                    csvrecord.append(name)
                    csvrecord.append('blur')
                    csvrecord.append(params)


                    csvrecord.append(result_before)
                    csvrecord.append(result_after)
                    print(csvrecord)

                   
                    writer.writerow(csvrecord)
        print("blur done")


                
        print("all done")



if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../Dataset',
                        help='path for dataset')
    args = parser.parse_args()
    rambo_testgen_coverage(args.dataset)

