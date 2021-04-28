import cv2
import os
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from watershed1 import *
#a=cv2.imread("/home/rakshita")
rect=(50,50,450,290)
bgdModel=np.zeros((1,65),np.float64)
fgdModel=np.zeros((1,65),np.float64)
ax=[]
images=[]
fig=plt.figure(figsize=(24,24))
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
def masking_algo(imag):
        mask=np.zeros(imag.shape[:2],np.uint8)
        cv2.grabCut(imag,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
        maskimg=imag*mask2[:,:,np.newaxis]
        cvtimg=cv2.cvtColor(maskimg,cv2.COLOR_BGR2GRAY)
        watershed_algo(cvtimg)
        return cvtimg
target=[]
data=np.empty([0,64*64*3])
b=os.listdir("/home/rakshita/pumpkindiseased")
for cname in b:
        iname=os.listdir("/home/rakshita/pumpkindiseased/"+cname)
        for img in iname:
            print("filename",img)
            c=cv2.imread("/home/rakshita/pumpkindiseased/"+cname+"/"+img)
            print("shape of the gray image",c.shape[0])
            if c.shape[0]>64:
                c=cv2.resize(c,(64,64))
                images.append(c)
            rcimg=masking_algo(c)
            print("shape:",c.shape)
            rcimg=c.reshape(1,64*64*3)
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
train_samples=int(data.shape[0]*0.8)
X_train=data[:train_samples,:]
X_test=data[train_samples:,:]
y_train=a[:train_samples]
y_test=a[train_samples:]
print(X_train)
print(X_test)
print(y_train)
print(y_test)
#model=SVC(kernel="linear",C=100)
#model=DecisionTreeClassifier()
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
confusedimages=[]
actvals=[]
predvals=[]
for i, y_p in enumerate(y_pred):
    print(y_p,":",y_test[i])
    if y_p!=y_test[i]:
        #print(X_test[i].reshape(64,64,3))
        confusedimages.append(images[i])
        actvals.append(y_test[i])
        predvals.append(y_p)
print("score=",model.score(X_test,y_test))
show_images(confusedimages,2)
b=input("enter the path")
#print(type(b))
c=cv2.imread(b)
if c.shape[0]>64:
	c=cv2.resize(c,(64,64))
	#print(type(c))
d=masking_algo(c)
d=c.reshape(1,64*64*3)
crop=model.predict(d)
predi=rle.inverse_transform(crop)
print(predi)
#plt.show()



