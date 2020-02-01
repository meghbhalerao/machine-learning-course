#!/usr/bin/env python
# coding: utf-8

# # Question 2
# The question asks us to capture images of alphabets, then binarize them and resize them to 9x7. This was done for the alphabets 'A', 'B', 'C', 'D', 'E', and 'F'. This was achieved using the OpenCV library available with Python. Functions used:
# * cv2.imread(): Reads image into a matrix
# * cv2.cvtColor(): Converts to the necessary colour format. In this question, we convert the original image to grayscale. In a gray scale image the values of the pixels lie between 0 and 255. 0 corresponds to black and 255 corresponds to white. 
# * cv2.threshold(): Converts the grayscale image to a black and white image. This functions performs binarization. We have used a threshold of 100 to obtain the binarized image. If the pixel value is greater than the threshold, then that pixel is assigned a value of 255 i.e white, otherwise the pixel is assigned black or value 0.
# * cv2.resize(): Resizes the binarized image to the required dimensions. This functions also performs interpolation to find out the values of pixels in the resized image. We have used the "Inter Area Interpolation" which is recommended for shrinking the image.
# 
# After the images are resized, the image in displayed and its pixel values are printed as a matrix with dimensions 9x7. Various observations were made.
# 

# In[1]:


import cv2
import os
width = 7
height = 9
dim = (width, height)
list_alpha = ['A','B','C','D','E','F']
for alpha in list_alpha:
    img = cv2.imread('F:\\notes\\8th sem\\Machine Learning-Jupyter Codes\\images\\'+alpha+'_1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    (thresh, bw) = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    img = cv2.resize(bw, dim,interpolation = cv2.INTER_AREA)
    cv2.imshow('',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('alphabet '+ alpha+' after resizing\n')
    print(img)
    print('\n')
    


# #### Observations
# * After resizing, the image was printed as a matrix of dimension 9x7.
# * The values of pixels in each image ranges from 240 to 255. To the naked eye, the image will appear to be entirely white.
# * The image after resizing is not binary because the cv2.resize() function performs interpolation. If we don't give any interpolation input to the cv2.resize() function, all the pixels become white.
# * This method cannot be used to generate images of alphabets having size 9x7. We will have to use bigger images. 

# In[ ]:




