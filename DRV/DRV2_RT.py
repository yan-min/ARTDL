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

from extract_vgg_layer import FeatureVisualization
from autumn_produce_back import produce
from autumn_produce_back import make_predictor
#os.environ['KERAS_BACKEND']= 'tensorflow'

def image_translation(img, params):
    rows,cols,ch = img.shape

    M = np.float32([[1,0,params[0]],[0,1,params[1]]])
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
    
    
    
    
    
if __name__ == "__main__":
    average_F_meatures = []
    model = make_predictor()
    now_time = time.strftime("%Y%m%d-%H%M%S",time.localtime())
    with open('./ART_result_test/DRV2_RT'+now_time+'.csv', 'wb',0) as csvfile:
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

            seed_inputs2 = "../Dataset/Ch2_001/center/"
            seed_labels2 = "../Dataset/Ch2_001/CH2_final_evaluation.csv"

            filelist2 = []
            for file in sorted(os.listdir(seed_inputs2)):
                if file.endswith(".jpg"):
                    filelist2.append(file)

            input_domain = []
            for file in filelist2:
                for i in range(1,11):
                    name = 'translation'+'_'+str(i)+'_'+str(file)
                    input_domain.append(name)
                for i in range(1,11):
                    name = 'brightness'+'_'+str(i)+'_'+str(file)
                    input_domain.append(name)
                for i in range(1,13):
                    name = 'motionBlur'+'_'+str(i)+'_'+str(file)
                    input_domain.append(name)
                for i in range(1,11):
                    if i != 5 and i != 8:
                        name = 'blur'+'_'+str(i)+'_'+str(file)
                        input_domain.append(name)



        # caculate F-meatures, the counts of failures
            F_meatures = 1
            error_count = 1
            reveal_failure = False
            while reveal_failure == False:
                random_ways = random.sample(input_domain,k=1)
                candidate_set = gen_candidate_set(random_ways)
                best_test = random.sample(candidate_set,k=1)[0]
                print(best_test)
    #             if best_test in failure_set:
                
                predict_after = produce(best_test,model)
                print(best_test.split("_")[-1])
                predict_before = produce(best_test.split("_")[-1],model)
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


