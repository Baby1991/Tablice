import cv2
import matplotlib.pyplot as plt
import pytesseract
import numpy as np
import text_detection
import os
from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def tablica(imgin):
    img= imgin.copy()
    prikaz=img.copy()
    img= cv2.normalize(img,img, 0, 255, cv2.NORM_MINMAX)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gauss=cv2.GaussianBlur(img,(5,5),0)
    gaussgray=cv2.GaussianBlur(gray,(5,5),0)

    treshgaussgray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    kernel = np.ones((5,5),np.uint8)
    
    gaussCanny=cv2.Canny(gauss,100,200,10)
    gaussCannygray=cv2.Canny(gaussgray,0,200,10)

    overlap=cv2.bitwise_and(gaussCanny,gaussCannygray)

    textboxes=[]
    fortextdetection=cv2.cvtColor(treshgaussgray,cv2.COLOR_GRAY2BGR)
    textboxes=text_detection.text_detection(fortextdetection,0.999)

    
    for (sx,sy,ex,ey) in textboxes:
        
        cv2.rectangle(prikaz,(sx,sy),(ex,ey),(0,0,0),3)
        test=img[sy:ey,sx:ex]
        text=pytesseract.image_to_string(test)
        if(text is not ""):
            print(text)

    (contours,__)=cv2.findContours( overlap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    height,width = img.shape[:2]
    blank_image = np.zeros((height,width,3), np.uint8)
    blank_image=cv2.cvtColor(blank_image,cv2.COLOR_BGR2GRAY)
    cv2.drawContours(blank_image,contours,-1,(255,255,255),1)
    blank_image=cv2.cvtColor(blank_image,cv2.COLOR_GRAY2RGB)
    

    for cnt in contours:
        A=0
        skor=0
        x,y,w,h = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(255,0,0),2)
        #if 2<=(w/h)<=7:
        cv2.rectangle(prikaz,(x,y),(x+w,y+h),(0,255,0),3)

        

    #plt.figure()
    #plt.imshow(overlap)
    
    plt.figure()
    plt.imshow(prikaz) 

    plt.figure()
    plt.imshow(blank_image) 

    plt.figure()
    plt.imshow(gaussCannygray) 

    plt.figure()
    plt.imshow(treshgaussgray)
    
    plt.show()





for filename in os.listdir("Slike"):
    if filename.endswith(".jpg"):
        img=cv2.imread("Slike/"+filename)
        tablica(img)