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


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print('')

def endtoend(iteracija, granica):
    brojac = 0
    iou = []
    tpr = []
    fpr = []
    ukupno = len(os.listdir(os.path.join(link, '')))/2
    print_progress_bar(0, 100, prefix=("\t" + os.path.join(link, '') + "\t"))
        
    for filename in os.listdir(os.path.join(link, '')):
        if filename.endswith(".txt"):
            f = open(os.path.join(link, f"{filename}"), "r")
            txt = f.read().split('\t')
            fajl = txt[0]
            img = cv2.imread(os.path.join(link, f"{fajl}"))
            name = fajl.split('.')[0]
            sx = int(txt[1])
            sy = int(txt[2])
            w = int(txt[3])
            h = int(txt[4])
            ex = sx+w
            ey = sy+h

            textboxes=[]
            textboxes = text_detection.text_detection(img, granica)        
            brojac += 1
            baza = Rectangle(sx, sy, ex, ey)
            povrsBaza = (ex-sx)*(ey-sy)
            height, width = img.shape[:2]
            povrsSlika = width*height
            
            presek=0
            povrsDetekt=0
            TP=0
            
            for (sx1,sy1,ex1,ey1) in textboxes:
                detektovano = Rectangle(sx1, sy1, ex1, ey1)
                povrsDetekt += (ex1-sx1)*(ey1-sy1)
                TP += area(baza, detektovano)
            
            FN = povrsBaza-TP
            FP = povrsDetekt-TP
            TN = povrsSlika-TP-FN-FP
            
            TPR = TP/(TP+FN)*100
            FPR = FP/(FP+TN)*100
            IOU = TP/(FP+TP+FN)*100
        
            iou.append(IOU)
            tpr.append(TPR)
            fpr.append(FPR)
        
            dosad = time.time()-start
            prosecnovreme = dosad/len(iou)
            preostalovreme = (ukupno-brojac) * prosecnovreme
            if preostalovreme >= 60:
                preostalovreme = str(round(preostalovreme/60, 1)) + " min"
            else:
                preostalovreme = str(round(preostalovreme, 1))+" s"
            print_progress_bar(brojac, ukupno, prefix=("\t"+link+"\t"+str(iteracija)+"\t"), suffix=("\t"+preostalovreme+"\t"+str(granica)+"\t"))
    return(iou, tpr, fpr)




text_file = open("Rezultati.txt", "w")
IOU=[]
TPR=[]
FPR=[]
link = os.path.join('..', 'benchmarks', 'endtoend', 'eu')
text_file.write(link+'\n')
text_file.flush()

first=1
last=9

for iteracija in range(first,last+1,1):
    granica=round(1/(last-first)*iteracija,2)
    start = time.time()
    iou, tpr, fpr = endtoend(iteracija,granica)
    vreme = time.time()-start

    _iou = sum(iou)/len(iou)
    IOU.append(_iou)
    _tpr = sum(tpr)/len(tpr)
    TPR.append(_tpr)
    _fpr = sum(fpr)/len(fpr)
    FPR.append(_fpr)

    text_file.write("\n")
    prosecnovreme = vreme/len(iou)

    #plt.figure()
    #plt.hist(iou, bins='auto')
    #plt.savefig(os.path.join('..', 'Rezultati', f'Rezultati_{iteracija}_histogram.jpg'))

    text_file.write(str(round(_iou, 2))+"\t"+str(round(_tpr, 2))+"\t" + str(round(_fpr, 2))+"\t"+str(len(granica))+"\t"+str(iou)+"\t\n")
    text_file.write("Prosecno vreme po slici: "+str(round(prosecnovreme, 2)
                                                    )+" s ("+str(round(vreme/60, 2))+" min)\n")
    text_file.flush()

_IOU=sum(IOU)/len(IOU)
text_file.write("\n"+str(round(_IOU,2)))
text_file.close()
plt.figure()
plt.plot(range(0,len(TPR)),TPR)
plt.savefig(os.path.join('..', 'Rezultati', 'TPR.jpg'))
plt.figure()
plt.plot(range(0,len(FPR)),FPR)
plt.savefig(os.path.join('..', 'Rezultati', 'FPR.jpg'))
plt.figure()
plt.plot(FPR,TPR)
plt.savefig(os.path.join('..', 'Rezultati', 'ROC.jpg'))
