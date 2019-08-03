import cv2 as cv
import cv2
import matplotlib.pyplot as plt
import pytesseract
import numpy as np

import os

#for filename in os.listdir("Slike"):
for i in range(1,2): 
    orig=cv.imread("Slike/YOO-5657.jpg")
    #orig=cv.imread("Slike/{0}".format(filename),1)
        
    #img=cv.threshold(img,127,255,1)
    #img=cv.blur(img,(10,10))
    img=orig
    hist,bins = np.histogram(img.flatten(),256,[0,256]) 
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[img]

    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    meanbright,_,_,_=cv.mean(img2)
    thres1, img21 = cv2.threshold(img2, meanbright, 255, 3)

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    meanbright,_,_,_=cv.mean(img)
    thres1, img1 = cv2.threshold(img2, meanbright, 255, 3)
    img1=cv.Canny(img1,100,200,10)


    #img21=cv.Canny(img21,100,200,10)
    #img21=cv.fastNlMeansDenoising(img21)

    #img3=cv.add(img1,img11)
    #img3=cv.bitwise_and(img1,img11)
    #img3=cv.fastNlMeansDenoising(img3)
    img3=img1
    #plt.imshow(img21)  
    #plt.show()
    #plt.imshow(img1)  
    #plt.show()
    
    img3=cv.blur(img3,(3,3))
    img3 = cv2.erode(img3, (5,5),10)

    
    #plt.imshow(img3)  
    #plt.show()
    




    

    (contours,__)=cv.findContours( img3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);
    (x1,y1,w1,h1)=(0,0,0,0)
    prikaz=orig

    dobrekonture=[]
    
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        tester=orig[y:y+h,x:x+w]
        tester=cv.cvtColor(tester, cv.COLOR_BGR2GRAY)
        (meanbright1,__,__,__)=cv.mean(tester)
        
        if 3<=(w/h)<=4.7:
        #if 1==1: 
            cv2.rectangle(prikaz, (x,y), (x+w,y+h), (0,255,0), 2)
            if meanbright1>=meanbright and w>150:
                cv2.rectangle(prikaz, (x,y), (x+w,y+h), (255,0,0), 2)
                dobrekonture.append(contour)
                #if(w>w1 and h>h1):
                #   (x1,y1,w1,h1)=(x,y,w,h)
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    imgbw=cv.cvtColor(orig,cv.COLOR_BGR2GRAY)
    #cv2.rectangle(orig, (x1,y1), (x1+w1,y1+h1), (0,0,0), 2)
    for contour in dobrekonture:
        (x,y,w,h) = cv2.boundingRect(contour)
        index=dobrekonture.index(contour)
        cv2.putText(prikaz,str(index),(x,y), font, 1,(0,0,0),2,cv2.LINE_AA)
        tester=imgbw[y:y+h,x:x+w]
        #color = ('b','g','r')
        color=('b')
        plt.figure()
        for i,col in enumerate(color):
            histr = cv2.calcHist([tester],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
            plt.legend(str(index))
        #plt.show()

    plt.figure()
    plt.imshow(prikaz) 
    plt.show()
    """
    orig=orig[y1:y1+h1,x1:x1+w1]
    #print(pytesseract.image_to_string(img))
    orig = cv.cvtColor(orig, cv.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(orig) 
    plt.show()
    """