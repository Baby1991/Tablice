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


def Povrsina(pravougaonik):
    return abs(pravougaonik[2] - pravougaonik[0]) * abs(pravougaonik[3] - pravougaonik[1])


def area(a, b):
    cetiritacke = [[a[0], a[1]], [a[2], a[3]],
                    [b[0], b[1]], [b[2], b[3]]]
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


def intersection_coordinates(pravougaonik1, pravougaonik2):
    # vraca dve koordinate pravougaonika preseka
    niz_x_koordinata = [pravougaonik1[0], pravougaonik1[2],
                        pravougaonik2[0], pravougaonik2[2]]

    niz_y_koordinata = [pravougaonik1[1], pravougaonik1[3],
                        pravougaonik2[1], pravougaonik2[3]]

    niz_x_koordinata.sort()
    niz_y_koordinata.sort()

    return (niz_x_koordinata[1], niz_y_koordinata[1], niz_x_koordinata[2], niz_y_koordinata[2])


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    #if iteration == total:
    #    print('')

def pripadanje_piksela(x, y, lista):
    for pravougaonik in lista:
        if x >= pravougaonik[0] and x <= pravougaonik[2] and y >= pravougaonik[1] and y <= pravougaonik[3]:
            return 1
    return 0

def endtoend(iteracija, granica):
    brojac = 0
    iou = []
    tpr = []
    fpr = []
    ukupno = len(os.listdir(os.path.join(link, '')))/2
    print_progress_bar(0, 100, prefix=(
        "\t"+link+"\t"+str(iteracija)+"\t"), suffix=("\t∞\t"+str(granica)+"\t"))

    for filename in os.listdir(os.path.join(link, '')):
        if filename.endswith(".txt"):
            f = open(os.path.join(link, f"{filename}"), "r")
            txt = f.read().split('\t')
            fajl = txt[0]
            img = cv2.imread(os.path.join(link, f"{fajl}"))
            sx = int(txt[1])
            sy = int(txt[2])
            w = int(txt[3])
            h = int(txt[4])
            ex = w
            ey = h

            textboxes = []
            textboxes = text_detection.text_detection(img, granica)

            brojac += 1
            baza = Rectangle(sx, sy, ex, ey)
            povrsBaza = (ex-sx)*(ey-sy)
            height, width = img.shape[:2]
            povrsSlika = width*height
            povrsDetekt = 0
            TP = 0

            
            textboxes1=[]
            for (sx1,sy1,ex1,ey1) in textboxes:
            
                sx1=max(0,sx1)
                sy1=max(0,sy1)
                ex1=max(0,ex1)
                ey1=max(0,ey1)

                sx1=min(width,sx1)
                sy1=min(height,sy1)
                ex1=min(width,ex1)
                ey1=min(height,ey1)
                
                textboxes1.append((sx1,sy1,ex1,ey1))
            
            textboxes=textboxes1.copy()
            
            # lista preseka zutih pravougaonika
            
            preseci_zutih = []
            i = 0
            for i in range(i, len(textboxes)):
                j = i + 1
                for j in range(j, len(textboxes)):
                    if area(textboxes[i], textboxes[j]) > 0:
                        preseci_zutih.append(intersection_coordinates(
                            textboxes[i], textboxes[j]))
            
            #ovo radi
            for pravougaonik in textboxes:
                povrsDetekt += Povrsina(pravougaonik)
            for presek in preseci_zutih: 
                povrsDetekt -= Povrsina(presek)
            
            baza_x = baza[0]
            baza_y = baza[1]
        
            while baza_x <= baza[2]:
                while baza_y <= baza[3]:
                    TP += pripadanje_piksela(baza_x, baza_y, textboxes)
                    #print(baza_x)
                    #print(baza_y)
                    baza_y += 1
                baza_y = baza[1]
                baza_x += 1

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
                preostalovreme = str(int(preostalovreme/60)) + " min"
            else:
                preostalovreme = str(int(preostalovreme))+" s"
            print_progress_bar(brojac, ukupno, prefix=(
                "\t"+link+"\t"+str(iteracija)+"\t"), suffix=("\t"+preostalovreme+"\t"+str(granica)+"\t"))
                         
    return(iou, tpr, fpr)


text_file = open("Rezultati.txt", "w")
IOU = []
TPR = []
FPR = []
link = os.path.join('..', 'MegaCrkotina')
text_file.write(
    link+'\t'+str(int(len(os.listdir(os.path.join(link, '')))/2))+'\n')
text_file.flush()

first = 0
last = 100
delta=1/(last-first)
program_start = time.time()
for iteracija in range(first, last+1, 1):
    granica = round(delta*iteracija, 2)
    start = time.time()
    iou, tpr, fpr = endtoend(iteracija, granica)
    vreme = time.time()-start

    _iou = sum(iou)/len(iou)
    IOU.append(_iou)
    _tpr = sum(tpr)/len(tpr)
    TPR.append(_tpr)
    _fpr = sum(fpr)/len(fpr)
    FPR.append(_fpr)

    text_file.write("\n")
    prosecnovreme = vreme/len(iou)

    text_file.write(str(iteracija)+'\t'+str(round(_iou, 2))+"\t"+str(round(_tpr, 2))+"\t" +
                    str(round(_fpr, 2))+"\t"+str(granica)+"\t\n")
    text_file.write("Prosecno vreme po slici: "+str(round(prosecnovreme, 2)
                                                    )+" s ("+str(round(vreme/60, 2))+" min)\n")
    text_file.flush()

print('\n')
program_run_time = time.time()-program_start
sati = program_run_time/60/60
minuti = program_run_time/60 % 60
sekunde = program_run_time % 60

_IOU = sum(IOU)/len(IOU)
text_file.write("\n"+str(round(_IOU, 2))+"%\t"+str(int(program_run_time/(first+last)/len(os.listdir(os.path.join(link, '')))/2))+"\t"+str(int(sati)) +
                "h\t"+str(int(minuti))+"min\t"+str(int(sekunde))+"s\n")
text_file.close()

x=np.arange(0, 1 + delta, delta).tolist()

fig = plt.figure()
plt.plot(x, TPR)
fig.suptitle("TPR")
plt.ylim(0, 100)
plt.xlabel('Nivo sigurnosti detekcije OCR-a')
plt.ylabel('TPR [%]')
fig.savefig(os.path.join('..', 'Rezultati', 'TPR.svg'))

fig = plt.figure()
plt.plot(x, FPR)
fig.suptitle("FPR")
plt.ylim(0, 100)
plt.xlabel('Nivo sigurnosti detekcije OCR-a')
plt.ylabel('FPR [%]')
fig.savefig(os.path.join('..', 'Rezultati', 'FPR.svg'))

fig = plt.figure()
plt.plot(FPR, TPR)
fig.suptitle("ROC")
plt.ylim(0, 100)
plt.xlabel('FPR [%]')
plt.ylabel('TPR [%]')
fig.savefig(os.path.join('..', 'Rezultati', 'ROC.svg'))
