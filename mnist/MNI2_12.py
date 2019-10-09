import os
import csv
import time
import random
import numpy as np
import cv2
from keras.preprocessing.image import load_img, img_to_array
from skimage.exposure import rescale_intensity
from scipy import misc
from scipy.misc import imread, imresize, imsave
from scipy.stats import wasserstein_distance
import random
from keras.datasets import mnist

from extract_vgg_layer import FeatureVisualization
from keras.layers import Input
from skimage import exposure, img_as_float

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
# (_, _), (x_test, _) = mnist.load_data()

# x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

path='./mnist.npz'
f = np.load(path)

x_test = f['x_test']
f.close()

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
#     rows,cols,ch = img.shape

    M = np.float32([[1,0,params[0]],[0,1,params[1]]])
    dst = cv2.warpAffine(img,M,(28,28))
    return dst

def image_scale(img, params):

    res = cv2.resize(img,None,fx=params[0], fy=params[1], interpolation = cv2.INTER_CUBIC)
    return res

def image_shear(img, params):
#     rows,cols,ch = img.shape
    factor = params*(-1.0)
    M = np.float32([[1,factor,0],[0,1,0]])
    dst = cv2.warpAffine(img,M,(28,28))
    return dst

def image_rotation(img, params):
#     rows,cols,ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),params,1)
    dst = cv2.warpAffine(img,M,(28,28))
    return dst

def image_moveBlur(image, degree):
    image = np.array(image)
    angle = 45
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
 
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
 
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

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

def EuclideanDistances(A, B):
    BT = B.transpose()
    vecProd = np.dot(A,BT)
    SqA =  A**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0
    ED = np.sqrt(SqED)
    return ED

def EMDDistances(A, B):
    m = len(A)
    n = len(B)
    ED = np.zeros([m,n])
    for i in range(0,m):
        x = A[i]
        for j in range(0,n):
            y = B[j]
            d=wasserstein_distance(x,y)
            ED[i][j] = d
    return ED
def gen_candidate_set(random_ways):
    candidate_set = {}
    for random_way in random_ways:
        way = random_way.split("_")[0]
        p = int(random_way.split("_")[1])
        seed_image = random_way.split("_")[2]
        seed_image_path = os.path.join(file_seed_path, seed_image)
        seed_image = imread(seed_image_path).reshape(28,28,1)
        if way == "moveBlur":
            params = p
            gen_image = image_moveBlur(seed_image,params)
            candidate_set[random_way] = gen_image
        if way == "translation":
            params = [p*0.25, p*0.25]
            gen_image = image_translation(seed_image,params)
            candidate_set[random_way] = gen_image
        if way == "brightness":
            params = p
            gen_image = image_brightness(seed_image,params)
            candidate_set[random_way] = gen_image
        if way == "blur":
            params = p
            gen_image = image_blur(seed_image,params)
            candidate_set[random_way] = gen_image
#         if way == "scale":
#             params = [p*0.5+1, p*0.5+1]
#             gen_image = image_scale(seed_image,params)
#             candidate_set[random_way] = gen_image

    return candidate_set

# iteration_param=10: top10 max distance
def select_best_test(executed_matrix,candidate_set):
 
    best_distance = -1.0
    for key, candidate in candidate_set.items():
        filename = os.path.join(file_total_path,key)
#        print(filename)
        candidate=cv2.imread(filename)
#        print(type(candidate))
        if not type(candidate) == 'NoneType':
            feature5_3 = myClass5_3.save_feature_to_img(candidate)
            candidate_matrix = feature5_3.reshape(1,3136)
            dis=EuclideanDistances(candidate_matrix,executed_matrix[1:])
            dis = np.array(dis)
            matrix1 = np.mean(dis, axis=1, keepdims=True) 
            mean_dis = matrix1[0][0]
            if best_distance < mean_dis:
                best_test = key
                best_distance = mean_dis
    return best_test
    
    
    
    
    
if __name__ == "__main__":
    average_F_meatures = []
    now_time = time.strftime("%Y%m%d-%H%M%S",time.localtime())
    with open('./result/MNI2_12_'+now_time+'art.csv', 'wb',0) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['index', 'F-measure','mean','std','M'])
        M=10000
        for item in range(0,2000):
        #     generate candidate_set and executed_set
            file_seed_path = "./seeds/1/"
            file_error_path = "./gen_images/error/1/"
            file_correct_path = "./gen_images/correct/1/"
            file_total_path = "./gen_images/total/1/"
            failure_set = []
            executed_set = []
            candidate_set = []
            files=os.listdir(file_correct_path)
            files_executed = random.sample(files, k=400)
            for f in files_executed:
                executed_set.append(os.path.join(file_correct_path,f))


            filelist = []
            for file in sorted(os.listdir(file_seed_path)):
                if file.endswith(".jpg"):
                    filelist.append(file)

            input_domain = []
            for file in filelist:
                for i in range(1,10):
                    name = 'translation'+'_'+str(i)+'_'+str(file)
                    input_domain.append(name)
                for i in range(1,11):
                    name = 'brightness'+'_'+str(i)+'_'+str(file)
                    input_domain.append(name)
                for i in range(1,7):
                    name = 'moveBlur'+'_'+str(i)+'_'+str(file)
                    input_domain.append(name)
                for i in range(1,9):
                    name = 'blur'+'_'+str(i)+'_'+str(file)
                    input_domain.append(name)
    #             for i in range(1,11):
    #                 name = 'scale'+'_'+str(i)+'_'+str(file)
    #                 input_domain.append(name)



        # caculate F-meatures, the counts of failures
            F_meatures = 1
            error_count = 1
            failure_set = []
            for file in os.listdir("./gen_images/error/1"):
                failure_set.append(file)
            reveal_failure = False
            #   generate executed_matrix to caculate the distance
            myClass5_3=FeatureVisualization(12)
            #       the size of executed_set is k=10
            executed_set = random.sample(executed_set,10)
            executed_matrix = np.zeros([1,3136])
            for executed_file in executed_set:
                executed_image=cv2.imread(executed_file)
                feature5_3 = myClass5_3.save_feature_to_img(executed_image)
                matrix = feature5_3.reshape(1,3136)
                executed_matrix = np.concatenate((executed_matrix,matrix),axis=0)
    #             print(executed_matrix.shape)
            while reveal_failure == False:
        #       the size of candidate_set is k=10
                random_ways = random.sample(input_domain,k=5)
                candidate_set = gen_candidate_set(random_ways)
                best_test = select_best_test(executed_matrix,candidate_set)
                print(best_test)
                way = best_test.split("_")[0]
                p = best_test.split("_")[1]
                ids = int(best_test.split("_")[2][:-4])
                source_image = np.expand_dims(x_test[ids], axis=0)
                if way == "moveBlur":
                    params = int(p)
                    gen_best_test = image_moveBlur(source_image[0],params)
                if way == "translation":
                    p = float(p)
                    params = [p*0.25, p*0.25]
                    gen_best_test = image_translation(source_image[0],params)
                if way == "brightness":
                    params = int(p)
                    gen_best_test = image_brightness(source_image[0],params)
                if way == "blur":
                    params = int(p)
                    gen_best_test = image_blur(source_image[0],params)
                predict_after = np.argmax(model2.predict(gen_best_test.reshape(1,28,28,1))[0])
#                print(predict_after)
                if not predict_after == 1:
                    print(str(error_count)+" failure is "+str(best_test)+" F-meatures is "+str(F_meatures))
                    reveal_failure = True
                else:
                    print("This test data did't find any failure")
                    F_meatures +=1
                    input_domain.remove(best_test)
                    executed_set.append(best_test)
            average_F_meatures.append(F_meatures)
            mean = np.mean(average_F_meatures)
            std = np.std(average_F_meatures,ddof=1)
            M = (196 * std/(5*mean)) ** 2
            print(average_F_meatures)
            print("mean: "+str(mean))
            print("std: "+str(std))
            print("M: "+str(M))
            csvrecord = []
            csvrecord.append(item)
            csvrecord.append(F_meatures)
            csvrecord.append(mean)
            csvrecord.append(std)
            csvrecord.append(M)
            writer.writerow(csvrecord)

        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

