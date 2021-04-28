import numpy as np
from scipy import ndimage
import cv2
from watershed1 import *
from matplotlib import pyplot as plt

oimg=cv2.imread('index.jpeg')
fig=plt.figure(figsize=(9,13))
mask=np.zeros(oimg.shape[:2],np.uint8)
bgdModel=np.zeros((1,65),np.float64)
fgdModel=np.zeros((1,65),np.float64)
rect=(50,50,450,290)
cv2.grabCut(oimg,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
img=oimg*mask2[:,:,np.newaxis]
img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
watershed_algo(img1)
ax=[]
ax.append(fig.add_subplot(2,2,1))
ax[-1].set_title("original image")
plt.imshow(oimg)
ax.append(fig.add_subplot(2,2,2))
ax[-1].set_title("segemented image")
plt.imshow(img1)
plt.show()



img3=cv2.imread('trial6.jpg')
mask=np.zeros(img3.shape[:2],np.uint8)
bgdModel=np.zeros((1,65),np.float64)
fgdModel=np.zeros((1,65),np.float64)
rect=(50,50,450,290)
cv2.grabCut(img3,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
img=img3*mask2[:,:,np.newaxis]
img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
watershed_algo(img2)
ax.append(fig.add_subplot(2,2,3))
ax[-1].set_title("original image")
plt.imshow(img3)
ax.append(fig.add_subplot(2,2,4))
ax[-1].set_title("segemented image")
plt.imshow(img2)
plt.show()

#plt.imshow(img2),plt.colorbar(),plt.show()

