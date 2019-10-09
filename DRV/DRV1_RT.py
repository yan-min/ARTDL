import os
import csv
import time
import random
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from skimage.exposure import rescale_intensity
from scipy import misc
from scipy.misc import imread, imresize, imsave
from scipy.stats import wasserstein_distance
from collections import deque
import sys
import argparse
from extract_vgg_layer import FeatureVisualization
# from autumn.autumn_produce_back import produce

class Model(object):
    def __init__(self,
                 model_path,
                 X_train_mean_path):

        self.model = load_model(model_path)
        self.model.compile(optimizer="adam", loss="mse")
        self.X_mean = np.load(X_train_mean_path)
        self.mean_angle = np.array([-0.004179079])
        self.img0 = None
        self.state = deque(maxlen=2)

    def predict(self, img_path0, img_path1, img_path2):
        self.img0 = load_img(img_path0, grayscale=True, target_size=(192, 256))
        self.img0 = img_to_array(self.img0)
        img1 = load_img(img_path1, grayscale=True, target_size=(192, 256))
        img1 = img_to_array(img1)
    
        img = img1 - self.img0
        img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
        img = np.array(img, dtype=np.uint8) # to replicate initial model
        self.state.append(img)
        self.img0 = img1

        img1 = load_img(img_path2, grayscale=True, target_size=(192, 256))
        img1 = img_to_array(img1)
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

def image_motionBlur(image, degree):
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
        seed_image_path = os.path.join("../Dataset/Ch2_001/center/", seed_image)
        seed_image = imread(seed_image_path)
        if way == "original":
            candidate_set[random_way] = seed_image
        if way == "motionBlur":
            params = p
            gen_image = image_motionBlur(seed_image,params)
            candidate_set[random_way] = gen_image
        if way == "translation":
            params = [p*10, p*10]
            gen_image = image_translation(seed_image,params)
            candidate_set[random_way] = gen_image
        if way == "brightness":
            params = p * 10
            gen_image = image_brightness(seed_image,params)
            candidate_set[random_way] = gen_image
        if way == "blur":
            params = p
            gen_image = image_blur(seed_image,params)
            candidate_set[random_way] = gen_image
        if way == "scale":
            params = [p*0.5+1, p*0.5+1]
            gen_image = image_scale(seed_image,params)
            candidate_set[random_way] = gen_image
        
    return candidate_set

# iteration_param=10: top10 max distance
def select_best_test(executed_matrix,candidate_set):
 
    best_distance = -1.0
    for key, candidate in candidate_set.items():
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
    
    
def predict_final(path, model):
    if len(path)>25:
        seed_path = "./new0109/"
        way = path.split("_")[0]
        seed_path1 = os.path.join(seed_path,way)
    else:
        seed_path1 = "../Dataset/Ch2_001/center"
    
    filelist = sorted(os.listdir(seed_path1))
    file_index = filelist.index(path)
    if file_index > 1:         
        yhat = model.predict(os.path.join(seed_path1, filelist[file_index-2]),os.path.join(seed_path1, filelist[file_index-1]),           os.path.join(seed_path1, path))
        return yhat[0]
    else:
        return "-0.004179079"    
    
    
if __name__ == "__main__":
    model = Model("./final_model.hdf5", "./X_train_mean.npy") 
    average_F_meatures = []
    now_time = time.strftime("%Y%m%d-%H%M%S",time.localtime())
    with open('./ART_result/DRV1_RT'+now_time+'.csv', 'wb',0) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['index', 'F-measure','mean','std','M'])
        M=10000   
        for item in range(0,2000):
        #     generate candidate_set and executed_set
            file_correct_path = "../Dataset/Ch2_001/center/"
            correct_result_path = "./predict_result.csv"
            correct_result_dict = {}
            failure_set = []
            with open(correct_result_path) as f1:
                reader = csv.reader(f1)
                head_row = next(reader)
                for row in reader:
                    correct_result_dict[row[0]] = row[1]
            executed_set = []
            candidate_set = []
            files=os.listdir(file_correct_path)
            files_executed = random.sample(files, k=400)
            for f in files_executed:
                executed_set.append(os.path.join(file_correct_path,f))

            seed_inputs1 = "../Dataset/hmb3/"
            seed_labels1 = "../Dataset/hmb3/hmb3_steering.csv"
            seed_inputs2 = "../Dataset/Ch2_001/center/"
            seed_labels2 = "../Dataset/Ch2_001/CH2_final_evaluation.csv"
            filelist1 = []
            for file in sorted(os.listdir(seed_inputs1)):
                if file.endswith(".jpg"):
                    filelist1.append(file)

            filelist2 = []
            for file in sorted(os.listdir(seed_inputs2)):
                if file.endswith(".jpg"):
                    filelist2.append(file)

            input_domain = []
            for file in filelist2:
#                name = 'original'+'_0_'+str(file)
#                input_domain.append(name)
                for i in range(1,10):
                    name = 'translation'+'_'+str(i)+'_'+str(file)
                    input_domain.append(name)
                for i in range(1,11):
                    name = 'brightness'+'_'+str(i)+'_'+str(file)
                    input_domain.append(name)
                for i in range(1,7):
                     name = 'motionBlur'+'_'+str(i)+'_'+str(file)
                     input_domain.append(name)
                for i in range(1,11):
                     if i != 5 and i != 8:
                         name = 'blur'+'_'+str(i)+'_'+str(file)
                         input_domain.append(name)
#                for i in range(1,11):
#                    name = 'scale'+'_'+str(i)+'_'+str(file)
#                     input_domain.append(name)



        # caculate F-meatures, the counts of failures
            F_meatures = 1
            error_count = 1
            reveal_failure = False
#             #   generate executed_matrix to caculate the distance
#             myClass5_3=FeatureVisualization(12)
#             #       the size of executed_set is k=10
#             executed_set = random.sample(executed_set,10)
#             executed_matrix = np.zeros([1,3136])
#             for executed_file in executed_set:
#                 executed_image=cv2.imread(executed_file)
#                 feature5_3 = myClass5_3.save_feature_to_img(executed_image)
#                 matrix = feature5_3.reshape(1,3136)
#                 executed_matrix = np.concatenate((executed_matrix,matrix),axis=0)
    #             print(executed_matrix.shape)
            while reveal_failure == False:
        #       the size of candidate_set is k=10
                random_ways = random.sample(input_domain,k=1)
                candidate_set = gen_candidate_set(random_ways)
                best_test = random.sample(candidate_set,k=1)[0]
                print(best_test)
    #             if best_test in failure_set:
                predict_after = predict_final(best_test, model)
                predict_before = predict_final(best_test.split("_")[-1], model)
                print(predict_before)
                print(predict_after)
                if abs(float(predict_after)-float(predict_before)) >0.1:
                    print(str(error_count)+" failure is "+str(best_test)+" F-meatures is "+str(F_meatures)) 
                    reveal_failure = True
                else:
                    print("this test data did't find any failure")
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


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





