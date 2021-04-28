import cv2
import os
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from watershed1 import *
#a=cv2.imread("/home/rakshita")
target=[]
data=np.empty([0,64*64])
b=os.listdir("/home/rakshita/pumpkin")
for cname in b:
        iname=os.listdir("/home/rakshita/pumpkin/"+cname)
        for img in iname:
            print("filename",img)
            c=cv2.imread("/home/rakshita/pumpkin/"+cname+"/"+img)
            fig=plt.figure(figsize=(9,13))
            mask=np.zeros(c.shape[:2],np.uint8)
            bgdModel=np.zeros((1,65),np.float64)
            fgdModel=np.zeros((1,65),np.float64)
            rect=(50,50,100,100)
            #rect=(50,50,450,290)
            print(c)
            print(c.shape)
            cv2.grabCut(c,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
            mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
            img=c*mask2[:,:,np.newaxis]
            img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            watershed_algo(img1)
            print("shape of the gray image",c.shape[0])
            if c.shape[0]>64:
                c=cv2.resize(c,(64,64))
                print(type(c))
                print(c)
            rcimg=c.reshape(3,64*64)
            data=np.vstack([data,rcimg])
            print(data)
            target.append(cname)
            print(target)
print(data.shape)
print(len(target))
rle=preprocessing.LabelEncoder()
rle.fit(target)
print(target)
a=rle.transform(target)

X_train=data
y_test=a
model=LinearSVC()
model.fit(X_train,y_test)
b=input("enter the path")
print(type(b))
c=cv2.imread(b,0)
fig=plt.figure(figsize=(9,13))
mask=np.zeros(c.shape[:2],np.uint8)
bgdModel=np.zeros((1,65),np.float64)
fgdModel=np.zeros((1,65),np.float64)
rect=(50,50,450,290)
cv2.grabCut(c,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
img=c*mask2[:,:,np.newaxis]
img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
watershed_algo(img2)
if img.shape[0]>64:
	img=cv2.resize(img,(64,64))
	print(type(img))
d=img.reshape(3,64*64)
crop=model.predict(d)
print(rle.inverse_transform(crop))
plt.show()
