import random
import io
import os
import re
import cv2
import scipy.io as matio
import numpy as np
from PIL import Image
from PIL import ImageFilter as IF
import ipdb
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
import torch
from prepare_datas.transforms import * 

image_size=[224,224]
def default_loader(image_path):

    return Image.open(image_path).convert('RGB')

normalize = transforms.Normalize(mean=(0.486, 0.457, 0.407),std=(0.229, 0.224, 0.225))  #vit-transformer


def findValue(value, interval):
    min_num = np.min(value)
    temp = np.linspace(min_num, min_num + len(value), num=len(value), endpoint=True, dtype=np.int64)
    if len(np.where(temp != value)[0]) == 0:
        index = min_num + len(value) - 1
    else:
        flag = 1
        index = np.max(np.where(value - temp < interval))
        while True:
            flag += 1
            if index + 1 < len(value) and value[index + 1] - temp[index + 1] < interval + value[index] - temp[index]:
                new_interval = interval + value[index] - temp[index]
                index = np.max(np.where(value - temp < new_interval))
            else:
                break
    row_top, row_bottom = min_num, index + min_num
    return row_top, row_bottom

def maskImageCal(DenoisedImage, m, n, mask_filter_ratio):
    temp = np.array(DenoisedImage)
    mask = np.zeros((m, n), dtype=np.uint8)  # operate on temp and store value to image_filtered
    mask1 = np.zeros((m, n), dtype=np.uint8)  # operate on temp and store value to image_filtered
    # This filter tries to find out available patches
    patch_size = np.array([m // mask_filter_ratio, n // mask_filter_ratio])  # smallest patch size we want 3,3
    patch_size_radius = patch_size // 2  # radius of filter 1,1
    max_out = nn.MaxPool2d(kernel_size=[2, 2], stride=1)
    input = torch.from_numpy(temp.astype(float)).unsqueeze(0)
    output = max_out(input)
    value = output.squeeze().numpy()
    value = np.array(value)
    shape = np.where(value!=0)

    if len(shape[0])==0:
        bound1 = np.array([2, 220, 2, 220])
        
    else:
        # print(shape[0])
        row_top, row_bottom = findValue(np.unique(shape[0]), m // 2)
        # print(row_top, row_bottom)
        row_bottom += patch_size_radius[0] * 2 - 1
        vol_left, vol_right = np.min(np.where(value[row_top:row_bottom] != 0)[1]), np.max(
            np.where(value[row_top:row_bottom] != 0)[1])
        vol_right += patch_size_radius[1] * 2 - 2
        if row_top<row_bottom and vol_left<vol_right:
            
            bound1 = np.array([row_top, row_bottom, vol_left, vol_right])
        else:
            bound1 = np.array([2, 220, 2, 220])
            
    return bound1

#------------------------------------------------------------------------stage:video+obj+bg-------------------------------------------------------
class VideoFrameDataset_stage(torch.utils.data.Dataset):  # msrvtt and activitynet
    def __init__(self, dataset_indicator, data_path=None, transform=None, loader=default_loader,flag='train'):
#         num_segments=8
        frames_index = np.array([0,4,8,12,16,20,24,29])
        num_segments=frames_index.shape[0]
        """Method to initilaize variables."""
        if dataset_indicator == 'msrvtt':
            if flag == 'train':
#                 ipdb.set_trace()
                path_sample = data_path + 'KF-TrainValVideoframes'
                sal_sample = data_path + 'sal_msrvtt_train_data'
                with io.open(data_path + 'train_list.txt', encoding='utf-8') as file:
                    video_sample = file.read().split('\n')[:-1]
                
                with io.open(data_path + 'id_train_msrvtt_labels.txt', encoding='utf-8') as file:
                    labels = file.read().split('\n')[:-1]

            elif flag == 'test':
                path_sample = data_path + 'KF-TestVideoframes'
                sal_sample = data_path + 'sal_msrvtt_test_data'
                with io.open(data_path + 'test_list.txt', encoding='utf-8') as file:
                    video_sample = file.read().split('\n')[:-1]
                with io.open(data_path + 'id_test_msrvtt_labels.txt', encoding='utf-8') as file:
                    labels = file.read().split('\n')[:-1]


        if dataset_indicator == 'activitynet':
            if flag == 'train':
#                 ipdb.set_trace()
                path_sample = data_path+'TrainVideoframes-30'
                sal_sample = data_path + 'sal_anet_train_data'
                with io.open(data_path + 'activitynet_trainlist.txt', encoding='utf-8') as file:
                    video_sample = file.read().split('\n')[:-1]
                with io.open(data_path + 'activitynet_train_id_labels.txt', encoding='utf-8') as file:
                    labels = file.read().split('\n')[:-1]

            elif flag == 'test':
                path_sample = data_path+'TestVideoframes-30'
                sal_sample = data_path + 'sal_anet_test_data'
                with io.open(data_path + 'activitynet_testlist.txt', encoding='utf-8') as file:
                    video_sample = file.read().split('\n')[:-1]
                   
                with io.open(data_path + 'activitynet_test_id_labels.txt', encoding='utf-8') as file:
                    labels = file.read().split('\n')[:-1]
               
        videoframes_path = []
        
        for i in range(len(video_sample)): 
            videoframes_path.append([])
            video_path = os.path.join(path_sample, video_sample[i])
            list_frames = os.listdir(video_path)
            if '.ipynb_checkpoints' in list_frames:
                list_frames.remove('.ipynb_checkpoints')

            list_frames = np.array(list_frames)
            list_frames = list_frames.tolist()
            list_frames.sort(key=lambda i : int(re.search(r'(\d+)',i).group()))
            list_frames = np.array(list_frames)

            sub_list_frames = list_frames[frames_index]
                   
            for j in range(num_segments):
                image_path = os.path.join(video_path,sub_list_frames[j])
                videoframes_path[i].append(image_path)
        
        sal_path = []
        
        for i in range(len(video_sample)): 
            sal_path.append([])
            sal_video_path = os.path.join(sal_sample, video_sample[i])
            sal_list_frames = os.listdir(sal_video_path)
            if '.ipynb_checkpoints' in sal_list_frames:
                sal_list_frames.remove('.ipynb_checkpoints')

            sal_list_frames = np.array(sal_list_frames)
            sal_list_frames = sal_list_frames.tolist()
            sal_list_frames.sort(key=lambda i : int(re.search(r'(\d+)',i).group()))
            sal_list_frames = np.array(sal_list_frames)
                   
            for j in range(num_segments):
                sal_image_path = os.path.join(sal_video_path,sal_list_frames[j])
                sal_path[i].append(sal_image_path)

        self.labels = np.array(labels, dtype=int)
        self.dataset_indicator = dataset_indicator
        self.transform = transform
        self.loader = loader
        self.videoframes_path = videoframes_path
        self.sal_path = sal_path


   
    def __getitem__(self, index):
#         ipdb.set_trace()
        labels = self.labels[index]
        videoframes_path = self.videoframes_path[index]
        sal_path = self.sal_path[index]

        object_videoframes=[]
        bg_videoframes =[]
        videoframes = []
        
        for i in range(len(videoframes_path)):
#             ipdb.set_trace()
            image_path = videoframes_path[i]
            sal_image_path = sal_path[i]
            
            img = cv2.imread(image_path) 
            im_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            im_resized = np.array(im_resized, dtype=np.float32)  #224,224,3

            sal_img = cv2.imread(sal_image_path,0)
            sal_img = np.array(sal_img)

            #bg
            threshold = 200  
            binary_image = np.where(sal_img >= threshold, 0, 1) 
            binary_image = np.expand_dims(binary_image, axis=0)
            
            im_resized_1 = np.transpose(im_resized, (2, 0, 1))
            bg_imgs = im_resized_1 * binary_image 
            
# #           #into tensor
            bg_imgs = np.transpose(bg_imgs,(1,2,0))
            bg_imgs = np.uint8(bg_imgs)
#             
        
            
            threshold2 = 10
            binary_image2 = np.where(sal_img >= threshold2, 255, 0)  
            binary_image2 = np.expand_dims(binary_image2, axis=0)
            bound = maskImageCal(binary_image2, 224, 224, 70)
    
            object_imgs = im_resized[bound[0]:bound[1], bound[2]:bound[3]] # h,w,3
            object_imgs = Image.fromarray(np.uint8(object_imgs))
#             print(type(object_imgs))
#             print(object_imgs.shape)
            object_imgs = object_imgs.resize(image_size, Image.BILINEAR) #224,224,3

            #to tensor
            object_imgs = transforms.ToTensor()(object_imgs) #3,224,224
            object_imgs = normalize(object_imgs)  #3,224,224 
            object_imgs = object_imgs.unsqueeze(0)
            object_videoframes.append(object_imgs)
            
            bg_imgs = transforms.ToTensor()(bg_imgs) #3,224,224
            bg_imgs = normalize(bg_imgs)  #3,224,224 
            bg_imgs = bg_imgs.unsqueeze(0)
#             print(bg_imgs.shape) #1,1,3,224,224
            bg_videoframes.append(bg_imgs)
            
            
#            
            r_img = Image.open(image_path).convert('RGB') #240,320,3
            r_img = r_img.resize(image_size, Image.BILINEAR) #224,224,3
            r_img = transforms.ToTensor()(r_img)
            r_img = normalize(r_img)  #3,224,224 
            r_img = r_img.unsqueeze(0)
            videoframes.append(r_img)
    
        videoframes=torch.cat(videoframes)
        object_videoframes = torch.cat(object_videoframes)
        bg_videoframes = torch.cat(bg_videoframes)
        
        if self.transform is not None:
            videoframes = transform(videoframes)
            object_videoframes=transform(object_videoframes)
            bg_videoframes=transform(bg_videoframes)
        return [videoframes,object_videoframes,bg_videoframes], labels


    def __len__(self):
        return len(self.videoframes_path)
    
 #-------------------------------------------------build dataset-------------------------------------------------------------------------------   

def build_dataset(train_stage, image_path, data_path, transform, mode, dataset_indicator,flag):
    dataset = VideoFrameDataset_stage(dataset_indicator,data_path=data_path, transform=None,loader=default_loader,flag=flag)
    return dataset
