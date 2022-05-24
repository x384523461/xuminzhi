import os.path
import matplotlib.pyplot as plt
import numpy as np
from skimage import io,transform
# import dataset
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class FaceLandmarksDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform = None):
        self.landmarks_frame=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform=transform
    def __len__(self):
        return len(self.landmarks_frame)
    def __getitem__(self, idx):
        img_name=os.path.join(self.root_dir,self.landmarks_frame.iloc[idx,0])
        image=io.imread(img_name)
        landmarks=self.landmarks_frame.iloc[idx,1:]
        landmarks=np.array([landmarks])
        landmarks=landmarks.astype('float32').reshape(-1,2)
        sample={'image':image,'landmarks':landmarks}
        if(self.transform):
            sample=self.transform(sample)
        return sample
class Rescale(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size=output_size
    def __call__(self, sample):
        image,landmarks=sample['image'],sample['landmarks']
        h,w=image.shape[:2]
        if(isinstance(self.output_size,int)):
            if(h>w):
                new_h,new_w=self.output_size*h/w,self.output_size
            else:
                new_h,new_w=self.output_size,self.output_size*w/h
        else:
            new_h,new=self.output_size
        new_h, new = int(new_h), int(new_w)
        img=transform.resize(image,(new_h,new_w))
        landmarks=landmarks*[new_w/w,new_h/h]
        return{'image':img,'landmarks':landmarks}
class RandomCrop(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        if(isinstance(output_size,int)):
            self.output_size=(output_size,output_size)
        else:
            assert len(output_size)==2
            self.output_size=output_size
    def __call__(self, sample):
        image,landmarks=sample['image'],sample['landmarks']
        h,w=image.shape[:2]
        new_h,new_w=self.output_size
        top=np.random.randint(0,h-new_h)
        left=np.random.randint(0,w-new_w)
        image=image[top:top+new_h,left:left+new_w]
        landmarks=landmarks-[left,top]
        return{'image':image,'landmarks':landmarks}
class Totensor(object):
    def __call__(self, sample):
        image,landmarks=sample['image'],sample['landmarks']
        image=image.transpose((2,0,1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}
def show_landmarks(image,landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:,0],landmarks[:,1],s=10,marker='.',c='r')
    plt.pause(0.001)
scale=Rescale(256)
crop=RandomCrop(128)
composed=transforms.Compose([Rescale(256),RandomCrop(224)])  #封装为一个组合操作
face_dataset=FaceLandmarksDataset(csv_file='data/faces/faces/face_landmarks.csv',root_dir='data/faces/faces',transform=None)

fig=plt.figure()
sample=face_dataset[65]
for i,tsfrm in enumerate([scale,crop,composed]):  #执行顺序为scale,crop,composed
    transformed_sample=tsfrm(sample)
    ax=plt.subplot(1,3,i+1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)
plt.show()

