import cv2 as cv
import cv2
import matplotlib.pyplot as plt
import pytesseract
import numpy as np
import text_detection
from PIL import Image


import os

from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
text_file = open("Output.txt", "w")

def area(a, b):
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0


for filename in os.listdir("Slike"):
#for i in range(1,2): 
    #orig=cv.imread("Slike/IEB-2949.jpg")
    orig=cv.imread("Slike/{0}".format(filename),1)
    broj=os.listdir("Slike").index(filename)  
    text_file.write(str(broj)+"\n") 
    print(broj)

    img=orig.copy()
    prikaz=orig.copy()

    textboxes=[]
    textboxes=text_detection.text_detection(img,0.999)
    
    """
    plt.figure()
    plt.imshow(text)
    plt.show()
    """


    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)
    meanbright,_,_,_=cv.mean(img)


    img=cv.blur(img,(3,3))
    thres1, img1 = cv2.threshold(img, meanbright, 255, 0)

    #thres1, img1 = cv2.threshold(img, meanbright, 255, 3)

    #img1=cv.blur(img1,(3,3))

    img1=cv.Canny(img,100,200,10)


    
    
    img1 = cv2.dilate(img1, (3,3),1)
    img1 = cv2.erode(img1, (3,3),1)
    img1 = cv2.dilate(img1, (3,3),10)

    
    #img1=cv.blur(img1,(3,3))
    #img1 = cv2.erode(img1, (5,5),10)

    img1slika=Image.fromarray(img1)
    img1slika.save("C:\\Users\\T420\\Documents\\GitHub\\Tablice\\Rezultati\\"+str(broj)+"_edz.jpg")
    
    
    
    (contours,__)=cv.findContours( img1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);
    (x1,y1,w1,h1)=(0,0,0,0)
    

    dobrekonture=[]
    mogucetablice=[(0,0)]

    cv.drawContours(prikaz, contours, -1, (0,0,255), 1)

    for r in textboxes:
        cv.rectangle(prikaz, (r.xmin,r.ymin), (r.xmax,r.ymax), (255,255,0), 2)
    
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        overlay = prikaz.copy()

        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 1)
        alpha = 0.4 
        prikaz = cv2.addWeighted(overlay, alpha, prikaz, 1 - alpha, 0)

        #cv2.rectangle(prikaz, (x,y), (x+w,y+h), (0,255,0), 2)
        if 100<=w<=400 and 30<=h<=100 and 2.5<(w/h)<6:
            tester=orig[y:y+h,x:x+w]
            tester=cv.cvtColor(tester, cv.COLOR_BGR2GRAY)
            (meanbright1,__,__,__)=cv.mean(tester)
            #cv2.rectangle(prikaz, (x,y), (x+w,y+h), (0,255,0), 2)

            if 0<=(w/h)<=100:
                if meanbright1>=meanbright/3:
                    cv2.rectangle(prikaz, (x,y), (x+w,y+h), (255,0,0), 2)
                    dobrekonture.append(contour)
                else:
                    cv2.rectangle(prikaz, (x,y), (x+w,y+h), (0,255,255), 2)
                
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    imgbw=cv.cvtColor(orig,cv.COLOR_BGR2GRAY)
    
    try:
        for contour in dobrekonture:
            A=0
            score=0
            (x,y,w,h) = cv2.boundingRect(contour)
            index=dobrekonture.index(contour)
            cv2.putText(prikaz,str(index),(x,y), font, 1,(0,0,0),2,cv2.LINE_AA)
            tester=imgbw[y:y+h,x:x+w]
            bound=Rectangle(x,y,x+w,y+h)
            boundarea=w*h
            histr = cv2.calcHist([tester],[0],None,[256],[0,256])
            beli=sum(histr[128:255])
            crni=sum(histr[0:127])

            if crni>0:
                odnos=beli/crni
            else:
                odnos=0

            for r in textboxes:
                A=A+area(r,bound)/boundarea*100+area(r,bound)/((r.xmax-r.xmin)*(r.ymax-r.ymin))*200
            
            if odnos>5:
                odnos=0
            odstup=abs(4-(w/h))
            score=A*100+odnos*20+min((w-40)/5+(h-10)/5,50)+min(w*h/50,50)+min(50/odstup,50)
            cv2.putText(prikaz,str(odnos),(x+30,y), font, 0.8,(0,0,0),2,cv2.LINE_AA,)
            #if(odnos>=0.1):
            mogucetablice.append((score,contour))
            pass
    except:
        print("ovo je autizam")
    """
    plt.figure()
    plt.imshow(prikaz) 
    plt.show()
    """
    prikazslika=Image.fromarray(prikaz)
    prikazslika.save("C:\\Users\\T420\\Documents\\GitHub\\Tablice\\Rezultati\\"+str(broj)+"_detekcija.jpg")
    maxscore,tablica=mogucetablice[0]
    for (score,contour) in mogucetablice:
        if(score!=0):
            (x,y,w,h) = cv2.boundingRect(contour)
            tester=orig[y:y+h,x:x+w]
            if(score>maxscore):
                maxscore,tablica=score,contour
          
    


    #orig=cv.imread("Slike/{0}".format(filename),1)
    #orig=cv.imread("Slike/IEB-2949.jpg")
    orig = cv.cvtColor(orig, cv.COLOR_BGR2RGB)

    try:
        (x1,y1,w1,h1) = cv2.boundingRect(tablica)
        orig=orig[y1:y1+h1,x1:x1+w1]
        raise
    except:
        cv2.putText(orig,"OVO JE LOS PROGRAM!!!!",(10,300), font, 2,(255,0,0),10,cv2.LINE_AA)


    text=pytesseract.image_to_string(orig)
    print(text)
    text_file.write(text+"\n") 
    #orig=cv.cvtColor(orig,cv.COLOR_BGR2RGB)
    origslika=Image.fromarray(orig)
    origslika.save("C:\\Users\\T420\\Documents\\GitHub\\Tablice\\Rezultati\\"+str(broj)+"_tablica.jpg")
    #plt.figure()
    #plt.imshow(final) 
    #plt.show()
text_file.close()    