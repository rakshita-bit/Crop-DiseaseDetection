import cv2
import os
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
#a=cv2.imread("/home/rakshita")
target=[]
data=np.empty([0,64*64])
b=os.listdir("/home/rakshita/pumpkin")
for cname in b:
        iname=os.listdir("/home/rakshita/pumpkin/"+cname)
        for img in iname:
            print("filename",img)
            c=cv2.imread("/home/rakshita/pumpkin/"+cname+"/"+img,0)
            print("shape of the gray image",c.shape[0])
            if c.shape[0]>64:
                c=cv2.resize(c,(64,64))
                print(type(c))
            rcimg=c.reshape(1,64*64)
            data=np.vstack([data,rcimg])
            print(data)
            target.append(cname)
            print(target)
print("the shape:",data.shape)
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
if c.shape[0]>64:
	c=cv2.resize(c,(64,64))
	print(type(c))
d=c.reshape(1,64*64)
crop=model.predict(d)
print(rle.inverse_transform(crop))
plt.show()
