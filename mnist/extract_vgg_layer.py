import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models
import os
import random
import csv

def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


class FeatureVisualization():
    def __init__(self,selected_layer):
        self.selected_layer=selected_layer
        self.pretrained_model = models.vgg16(pretrained=True).features

    def get_feature(self,img_path):
#         img=cv2.imread(img_path)

        input=preprocess_image(img_path)
        # input = Variable(torch.randn(1, 3, 224, 224))
        x=input
        for index,layer in enumerate(self.pretrained_model):
            x=layer(x)
            if (index == self.selected_layer):
                return x

    def get_single_feature(self,img_path):
        features=self.get_feature(img_path)
        #(1, 128, 56, 56) pass the selected_layer,output 128 feature_map
        feature=features[:,0,:,:]

        feature=feature.view(feature.shape[1],feature.shape[2])

        return feature

    def save_feature_to_img(self,img_path):
        #to numpy
        feature=self.get_single_feature(img_path)
        feature=feature.data.numpy()

        #use sigmod to [0,1]
        feature= 1.0/(1+np.exp(-1*feature))

        # to [0,255]
        feature=np.round(feature*255)
        cv2.imwrite('./img.jpg',feature)
        return feature
#4_2*5_3*5_3T
#flattened
#store by key value such as blur matrix, translation matrix
if __name__=='__main__':
    jlist = [2,3,4,5,6,7,8,9]
    for j in jlist:
        print("==============================number"+str(j)+"=================================")
        ilist = [10,11,12,13,14,15,18,19,20]
        for i in range(0,31):
            type_matrix = []
            # get class
            myClass1_1=FeatureVisualization(i)
            img=cv2.imread("./source_mnist.jpg")
            feature5_3 = myClass1_1.save_feature_to_img(img)
            print(feature5_3.shape)
            m = feature5_3.shape[0] * feature5_3.shape[1]
            final_matrix = np.zeros([1,m])

            file_error = "./gen_images/error/"+str(j)+"/"
            file_correct = "./gen_images/correct/"+str(j)+"/"

            print("-----------------------correct"+str(i)+"-------------------------------")

            files=os.listdir(file_correct)
            files = random.sample(files, k=400)
            correct_matrix = np.zeros([1,m])
            for file in files:
                filename = os.path.join(file_correct,file)
                print(filename)
                img=cv2.imread(filename)
                feature5_3 = myClass1_1.save_feature_to_img(img)
        #             feature5_2 = myClass5_2.save_feature_to_img(img).reshape(1,3136)
        #             feature5_1 = myClass5_1.save_feature_to_img(img).reshape(1,3136)
        #         feature5_3_T = feature5_3.T
        #         feature_G = np.dot(feature5_3, feature5_3_T)
        #         feature_G = feature_G.reshape(1,56*56)
        #         feature4_2 = feature4_2.reshape(1,56*56)
        #         matrix=np.concatenate((feature4_2,feature_G),axis=1)
        #             matrix=np.concatenate((feature5_1,feature5_2,feature5_3),axis=1)
                matrix = feature5_3.reshape(1,m)
                final_matrix = np.concatenate((final_matrix,matrix),axis=0)
                correct_matrix = np.concatenate((correct_matrix,matrix),axis=0)
                type_matrix.append(0)
                print(final_matrix.shape)    

            print("-----------------------error"+str(i)+"-------------------------------")
            files=os.listdir(file_error)
            files = random.sample(files, k=400)
            error_matrix = np.zeros([1,m])
            for file in files:
                filename = os.path.join(file_error,file)
                print(filename)
                img=cv2.imread(filename)
                feature5_3 = myClass1_1.save_feature_to_img(img)
        #             feature5_2 = myClass5_2.save_feature_to_img(img).reshape(1,3136)
        #             feature5_1 = myClass5_1.save_feature_to_img(img).reshape(1,3136)
        #         feature5_3_T = feature5_3.T
        #         feature_G = np.dot(feature5_3, feature5_3_T)
        #         feature_G = feature_G.reshape(1,56*56)
        #         feature4_2 = feature4_2.reshape(1,56*56)
        #         matrix=np.concatenate((feature4_2,feature_G),axis=1)
        #             matrix=np.concatenate((feature5_1,feature5_2,feature5_3),axis=1)
                matrix = feature5_3.reshape(1,m)
                final_matrix = np.concatenate((final_matrix,matrix),axis=0)
                error_matrix = np.concatenate((error_matrix,matrix),axis=0)
                type_matrix.append(1)
                print(final_matrix.shape)
        #   use for pca
            np.save("./pca/"+str(j)+"/"+str(i)+"final_matrix.npy", final_matrix)
            type_matrix=np.array(type_matrix)
            np.save("./pca/"+str(j)+"/"+str(i)+"final_matrix_type.npy",type_matrix)
        #   use for cos_distance
            np.save("./pca/"+str(j)+"/"+str(i)+"error_matrix.npy", error_matrix)
        #     np.save("13seed_matrix.npy",seed_matrix)
            np.save("./pca/"+str(j)+"/"+str(i)+"correct_matrix.npy",correct_matrix)

