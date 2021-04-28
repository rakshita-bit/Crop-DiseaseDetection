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
import pickle
from pumpkin3 import *

loaded_model=pickle.load(open(filename,'rb'))
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
result=loaded_model.score(X_test,y_test)
print("score=",result)
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
print(rle.inverse_transform(crop))
#plt.show()
                                

