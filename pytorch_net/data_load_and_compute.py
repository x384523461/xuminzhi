from __future__ import print_function,division
import os
import torch
import pandas as pd
from skimage import io,transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
import warnings
warnings.filterwarnings("ignore")
plt.ion()


landmarks_frame=pd.read_csv('data/faces/faces/face_landmarks.csv')
n=65
img_name=landmarks_frame.iloc[n,0]
landmarks=landmarks_frame.iloc[n,1:].to_numpy()
landmarks=landmarks.astype('float32').reshape(-1,2)

print('Image name:{}'.format(img_name))
print('Landmarks shape:{}'.format(landmarks.shape))
print('First 4 Landmarks:{}'.format(landmarks[:4]))

def show_landmarks(image,landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:,0],landmarks[:,1],s=10,marker='.',c='r')
    plt.pause(0.001)
plt.figure()
show_landmarks(io.imread(os.path.join('data/faces/faces/', img_name)),landmarks)

plt.show()

