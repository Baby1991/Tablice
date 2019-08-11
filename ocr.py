import cv2 as cv
import cv2
import matplotlib.pyplot as plt
import pytesseract
import numpy as np
import text_detection
from PIL import Image
import time

import os

from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
#text_file = open("Output.txt", "w")


def area(a, b):
    cetiritacke = [[a.xmin, a.ymin], [a.xmax, a.ymax],
                   [b.xmin, b.ymin], [b.xmax, b.ymax]]
    x21 = cetiritacke[2][0]
    y21 = cetiritacke[2][1]
    x22 = cetiritacke[3][0]
    y22 = cetiritacke[3][1]

    x11 = cetiritacke[0][0]
    y11 = cetiritacke[0][1]
    x12 = cetiritacke[1][0]
    y12 = cetiritacke[1][1]

    if ((x21 > x11 and x21 > x12) and (x22 > x11 and x22 > x12)) or ((x21 < x11 and x21 < x12) and (x22 < x11 and x22 < x12)):
        return 0
    if ((y21 > y11 and y21 > y12) and (y22 > y11 and y22 > y12)) or ((y21 < y11 and y21 < y12) and (y22 < y11 and y22 < y12)):
        return 0

    arry = [y11, y12, y21, y22]
    arrx = [x11, x12, x21, x22]
    arry.sort()
    arrx.sort()
    return (arrx[2] - arrx[1]) * (arry[2] - arry[1])


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print('')


def merge(textbox1, textbox2):
    textbox3 = []
    if textbox1[0] < textbox2[0]:
        textbox3.append(textbox1[0])
        textbox3.append(textbox1[1])
        textbox3.append(textbox2[2])
        textbox3.append(textbox2[3])
        return tuple(textbox3)

    textbox3.append(textbox2[0])
    textbox3.append(textbox2[1])
    textbox3.append(textbox1[2])
    textbox3.append(textbox1[3])
    return tuple(textbox3)


def histogrami(img, sx, sy, ex, ey):
    img = img[sy:ey, sx:ex]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    avgHue = np.average(hue)/255*360
    avgSat = np.average(saturation)/255*100
    avgVal = np.average(value)/255*100

    hue = hue.flatten()/255*360
    saturation = saturation.flatten()/255*100
    value = value.flatten()/255*100

    return(avgHue, avgSat, avgVal)


def tablica(img,name, granica):
    pomocna = 1
    orig=img.copy()

    img = orig.copy()
    prikaz = orig.copy()
    textboxes = []
    
    textboxes = text_detection.text_detection(img, granica)
       
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)
    add = cv2.add(value, topHat)
    subtract = cv2.subtract(add, blackHat)
    blur = cv2.GaussianBlur(subtract, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

    CannyThresh = cv2.Canny(thresh, 0, 200, 10)

    """
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    meanbright,_,_,_=cv.mean(img)

    img=cv.blur(img,(5,5))
    thres1, img1 = cv2.threshold(img, meanbright, 255, 0)

    img1=cv.Canny(img,100,200,10)

    img1 = cv2.dilate(img1, (3,3),1)
    img1 = cv2.erode(img1, (3,3),1)
    img1 = cv2.dilate(img1, (3,3),10)
    """
    
    (contours, __) = cv.findContours(
        CannyThresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    (x1, y1, w1, h1) = (0, 0, 0, 0)

    dobrekonture = []
    mogucetablice = []

    cv.drawContours(prikaz, contours, -1, (0, 0, 255), 1)

    for (sx, sy, ex, ey) in textboxes:
        cv.rectangle(prikaz, (sx, sy), (ex, ey), (255, 255, 0), 2)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        overlay = prikaz.copy()

        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 1)
        alpha = 0.7
        prikaz = cv2.addWeighted(overlay, alpha, prikaz, 1 - alpha, 0)

        if 1.5 <= (w/h) <= 1000 and 1000 > w > 30 and 1000 > h > 5:

            cv2.rectangle(prikaz, (x, y), (x+w, y+h), (255, 0, 0), 2)
            dobrekonture.append(contour)

    font = cv2.FONT_HERSHEY_SIMPLEX
        
    skorovi = []
        
    for contour in dobrekonture:
        A=0
        score=0
        (x,y,w,h) = cv2.boundingRect(contour)
        bound=Rectangle(x,y,x+w,y+h)
        boundarea=w*h
        
        for (sx,sy,ex,ey) in textboxes:
            r=Rectangle(sx,sy,ex,ey)
            A += area(r,bound)/boundarea*100 + area(r,bound)/((ex-sx)*(ey-sy)) * 70
               
        score=A
        skorovi.append(score)
        
        cv2.putText(prikaz,str(round(score,0)),(x,y), font, 0.4,(0,0,0),2,cv2.LINE_AA,)
        
        mogucetablice.append((score,contour))
                

    if len(mogucetablice) > 0:
        maxscore, tablica = mogucetablice[0]
        for (score, contour) in mogucetablice:
            (x, y, w, h) = cv2.boundingRect(contour)

            if(score>maxscore):
                maxscore, tablica = score, contour

        
        
                
    
    orig = cv.cvtColor(orig, cv.COLOR_BGR2RGB)

    
    (x1,y1,w1,h1) = cv2.boundingRect(tablica)
    orig=orig[y1:y1+h1,x1:x1+w1]

    #text=pytesseract.image_to_string(orig)
    text=""

    return((x1, y1, x1+w1, y1+h1))

def endtoend(granica,iteracija,start):
    brojac = 0
    ukupno = len(os.listdir(link))/2
    printProgressBar(0, 100, prefix=("\t"+link+"\t"+str(iteracija)+"\t"+str(granica)+"\t"))
    iou=[]
    tpr=[]
    fpr=[]


    text_file.flush()
    for filename in os.listdir(link):
        if filename.endswith(".txt"):
            
            f = open(link+"{0}".format(filename), "r")
            txt = f.read().split('\t')
            fajl = txt[0]
            img = cv2.imread(link+"{0}".format(fajl))
            name = fajl.split('.')[0]
            sx = int(txt[1])
            sy = int(txt[2])
            w = int(txt[3])
            h = int(txt[4])
            ex = sx+w
            ey = sy+h
            (sx1, sy1, ex1, ey1)= tablica(img, name, granica)

            (avgHue, avgSat, avgVal) = histogrami(img, sx1, sy1, ex1, ey1)

            height, width = img.shape[:2]
            detektovano = Rectangle(sx1, sy1, ex1, ey1)
            baza = Rectangle(sx, sy, ex, ey)
            povrsBaza=(ex-sx)*(ey-sy)
            povrsDetekt=(ex1-sx1)*(ey1-sy1)
            povrsSlika=width*height
            TP = area(baza, detektovano)
            FN=povrsBaza-TP
            FP=povrsDetekt-TP
            TN=povrsSlika-TP-FN-FP
            TPR=TP/(TP+FN)
            FPR=FP/(FP+TN)
            IOU=TP/(FP+TP+FN)

            iou.append(IOU*100)
            tpr.append(TPR*100)
            fpr.append(FPR*100)

            brojac += 1
            dosad = time.time()-start
            prosecnovreme = dosad/brojac
            preostalovreme = (ukupno-brojac)*prosecnovreme
            if preostalovreme >= 60:
                preostalovreme = str(round(preostalovreme/60, 1))+" min"
            else:
                preostalovreme = str(round(preostalovreme, 1))+" s"
            printProgressBar(brojac, ukupno, prefix=(
                "\t"+link+"\t"+str(iteracija)+"\t"+str(granica)+"\t"), suffix=("\t"+preostalovreme+"\t\t"))
    return(iou,tpr,fpr)

def Program(granica, iteracija):
    start = time.time()
    iou,tpr,fpr=endtoend(granica,iteracija,start)
    vreme = time.time()-start

    _iou = sum(iou)/len(iou)
    _fpr = sum(fpr)/len(fpr)
    _tpr = sum(tpr)/len(tpr)
    
    prosecnovreme = vreme/len(iou)

    plt.figure()
    plt.hist(iou, bins='auto')
    plt.savefig('../Rezultati/Rezultati_histogram'+str(iteracija)+'.jpg')
    text_file.write("Granica: "+str(round(granica, 2))+"\n")
    text_file.write("IOU: "+str(round(_iou, 5))+"% TPR: "+str(round(_tpr, 5))+"%"+ " FPR:" +str(round(_fpr, 5))+" % ("+str(len(iou))+")\n")
    text_file.write("Prosecno vreme po slici: "+str(round(prosecnovreme, 2))+" s ("+str(round(vreme/60, 2))+" min)\n\n")
    text_file.flush()
    return(_iou,_fpr,_tpr)
    
text_file = open("Rezultati.txt", "w")
link = "../benchmarks/endtoend/eu/"    
text_file.write("Fajl:\t"+link+"\n\n")

iou=[]
fpr=[]
tpr=[]

prvi=0
poslednji=10
increment=2
odnos=poslednji-prvi+1

for i in range(prvi,poslednji+1,increment):
    IOU,FPR,TPR=Program(i/10,i)
    iou.append(int(IOU))
    fpr.append(int(FPR))
    tpr.append(int(TPR))

text_file.close()
plt.figure()
plt.scatter(fpr,tpr)
plt.savefig('../Rezultati/ROC.jpg')
