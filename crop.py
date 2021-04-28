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
b=os.listdir("/home/rakshita/crops")
for cname in b:
        iname=os.listdir("/home/rakshita/crops/"+cname)
        for iname1 in iname:
                c=os.listdir("/home/rakshita/crops/"+cname+"/"+iname1)	
                for img in c:
                        print("file name",img)
                        cimg=cv2.imread("/home/rakshita/crops/"+cname+"/"+iname1+"/"+img,0)
                        print("/home/rakshita/crops/"+cname+"/"+iname1+"/"+img)
                        print(cimg.size)
                        print(cimg)
                        print("shape of the gray image",cimg.shape[0])
                        if cimg.shape[0]>64:
                                cimg=cv2.resize(cimg,(64,64))
                                print(type(cimg))
                        rcimg=cimg.reshape(1,64*64)
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
if c.shape[0]>64:
	c=cv2.resize(c,(64,64))
	print(type(c))
d=c.reshape(1,64*64)
crop=model.predict(d)
print(rle.inverse_transform(crop))
plt.show()
