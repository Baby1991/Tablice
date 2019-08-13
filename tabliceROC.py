import cv2 as cv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import text_detection
from PIL import Image
import time
import os
from collections import namedtuple

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def obrada_blackhat(img):
    #Trenutni algoritam
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    top_hat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)
    add = cv2.add(value, top_hat)
    subtract = cv2.subtract(add, black_hat)
    blur = cv2.GaussianBlur(subtract, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9
    )
    canny_thresh = cv2.Canny(thresh, 0, 200, 10)
    return(canny_thresh)

def obrada_canny_overlap(img):
    #Eksperimentalni algoritam
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gauss=cv2.GaussianBlur(img,(5,5),0)
    gaussgray=cv2.GaussianBlur(gray,(5,5),0)
    gaussCanny=cv2.Canny(gauss,100,200,10)
    gaussCannygray=cv2.Canny(gaussgray,0,200,10)
    overlap=cv2.bitwise_and(gaussCanny,gaussCannygray)
    return(overlap)

def obrada_canny_stari(img):
    #Prvi algoritam
    img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    meanbright,_,_,_=cv.mean(img)
    img=cv.blur(img,(5,5))
    thres1, img1 = cv2.threshold(img, meanbright, 255, 0)
    img1=cv.Canny(img,100,200,10)
    img1 = cv2.dilate(img1, (3,3),1)
    img1 = cv2.erode(img1, (3,3),1)
    img1 = cv2.dilate(img1, (3,3),10)
    return(img1)

def Povrsina(pravougaonik):
    #Izracunavanje povrsine Rectangle objekta (namedTuple)
    return abs(pravougaonik[2] - pravougaonik[0]) * abs(pravougaonik[3] - pravougaonik[1])

def area(a: namedtuple, b: namedtuple) -> int:
    #Funkcija koja racuna povrsinu preseka dva pravougaonika
    
    cetiri_tacke = [
        [a.xmin, a.ymin],
        [a.xmax, a.ymax],
        [b.xmin, b.ymin],
        [b.xmax, b.ymax]
    ]
    x21 = cetiri_tacke[2][0]
    y21 = cetiri_tacke[2][1]
    x22 = cetiri_tacke[3][0]
    y22 = cetiri_tacke[3][1]

    x11 = cetiri_tacke[0][0]
    y11 = cetiri_tacke[0][1]
    x12 = cetiri_tacke[1][0]
    y12 = cetiri_tacke[1][1]

    # Provera postojanja preseka pravougaonika
    if (
            (x21 > x11 and x21 > x12) and
            (x22 > x11 and x22 > x12)
    ) or (
            (x21 < x11 and x21 < x12) and
            (x22 < x11 and x22 < x12)
    ):
        return 0

    if (
            (y21 > y11 and y21 > y12) and
            (y22 > y11 and y22 > y12)
    ) or (
            (y21 < y11 and y21 < y12) and
            (y22 < y11 and y22 < y12)
    ):
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
    #Iscrtava progress bar
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')

def pripadanje_piksela(x, y, lista):
    #Odredjuje da li pixel pripada nekom od pravougaonika u unetoj listi (namedTuple list)
    for pravougaonik in lista:
        if x >= pravougaonik[0] and x <= pravougaonik[2] and y >= pravougaonik[1] and y <= pravougaonik[3]:
            return 1
    return 0

def merge(textbox1: tuple, textbox2: tuple) -> tuple:
    #Spajanje dva pravougaonika (Tuple)

    textbox3: tuple = []
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
    
    #Vraca srednje vrednosti Hue Saturation i Value dela slike img definisanim koordinatama sx,sy,ex,ey
    
    img = img[sy:ey, sx:ex]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    avg_hue = np.average(hue)/255*360
    avg_sat = np.average(saturation)/255*100
    avg_val = np.average(value)/255*100

    return avg_hue, avg_sat, avg_val


def tablica(img: Image, name, granica ):
    #Izdvaja tablicu sa slike img
    #granica ==> Granica sigurnosti OCR-a

    # Flag za proveru postojanja preseka
    flag_presek = 1
    orig = img.copy()
    kopija = orig.copy()
    height, width = img.shape[:2]

    textboxes = []
    #Detekcija text-a na slici - Vraca listu tuplova koji sadrze koordinate detektovanih pravougaonika koji sadrze tekst
    textboxes = text_detection.text_detection(img, granica)

    #Svi pravougaonici kojima koordinate izlaze van slike se vracaju u sliku
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
        
    #prolazak kroz ocr pravougaonike i provera da li mogu nepotrebni da se izbace
    #ili da li mogu da se spoje zalejno ukoliiko su dovoljno blizu
    if len(textboxes) > 1:
        i = 0
        while i < len(textboxes):
            j = i + 1
            while j < len(textboxes):
                if (
                        abs(textboxes[i][2] - textboxes[j][0]) < 40 or
                        abs(textboxes[i][0] - textboxes[j][2]) < 40
                ) or (
                        abs(textboxes[i][3] - textboxes[j][1]) < 10 or
                        abs(textboxes[i][1] - textboxes[j][3]) < 10
                ):
                    textboxes.insert(i + 1, merge(textboxes[i], textboxes[j]))
                    textboxes.pop(i + 1)
                    textboxes.pop(j)
                j += 1
            i += 1

            
    #ALGORITMI OBRADE
    processing_out=obrada_blackhat(img)

    (contours, __) = cv.findContours(
        processing_out, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    #liste potencijalnih kandidata
    dobre_konture = []
    moguce_tablice = []
    #glavno filtriranje crvenih pravougaonika
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if 1.5 <= (w/h) <= 10 and 1000 > w > 100 and 1000 > h > 5:
            dobre_konture.append(contour)
    #glavni deo selekcije kandidata pomocu sistema skorovanja
    #prema preseku crvenih i zutih pravougaonika
    for contour in dobre_konture:
        A = 0
        score = 0
        (x, y, w, h) = cv2.boundingRect(contour)
        bound = Rectangle(x, y, x+w, y+h)
        boundarea = w * h

        for (start_x, start_y, end_x, end_y) in textboxes:
            r = Rectangle(start_x, start_y, end_x, end_y)
            prekrivena_povrs_texta = area(
                r, bound)/((end_x-start_x)*(end_y-start_y))
            prekrivena_povrs_konture = area(r, bound)/boundarea
            A = max(A, prekrivena_povrs_konture *
                    100 + prekrivena_povrs_texta * 100)

        moguce_tablice.append((A, contour))

    skorovi = []
    #vracanje kandidata sa najvecim skorom ili ukoliko isti ne
    #postoji rade se histogrami na HSV verziji slike i na osnovu
    #toga se vraca tablica
    if len(moguce_tablice) > 0:
        maxscore, tablica = moguce_tablice[0]
        for (score, contour) in moguce_tablice:
            (x, y, w, h) = cv2.boundingRect(contour)
            skorovi.append(score)
            if(score > maxscore):
                maxscore, tablica = score, contour

        minsat = 500
        pozicija = 0
        pozicija2 = 0
        if maxscore == 0:
            flag_presek = 0
            i = 0
            for crkotina in moguce_tablice:
                (x, y, w, h) = cv2.boundingRect(crkotina[1])
                pg = Rectangle(x, y, x+w, y+h)
                (_, sat, _) = histogrami(kopija, pg[0], pg[1], pg[2], pg[3])
                if sat < minsat:
                    minsat = sat
                    pozicija = i
                i += 1
            (minsat, tablica) = moguce_tablice[pozicija]

            minsat = 500
            k = 0
            for txt in textboxes:
                (_, sat, _) = histogrami(
                    kopija, txt[0], txt[1], txt[2], txt[3])
                if sat < minsat:
                    minsat = sat
                    pozicija2 = k
                k += 1
    #provera da li postoji bilo kakav presek izmedju pravougaonika
    #Ukoliko ne postoji -> histogrami i saturacija
    if flag_presek == 0 and len(textboxes)>0:
        (x1, y1, w1, h1) = cv2.boundingRect(tablica)
        (_, sat1, _) = histogrami(img, x1, y1, x1 + w1, y1 + h1)
        (_, sat2, _) = histogrami(
            img, textboxes[pozicija2][0], textboxes[pozicija2][1], textboxes[pozicija2][2], textboxes[pozicija2][3])

        if sat2 >= sat1:
            (x1, y1, w1, h1) = tuple(textboxes[pozicija2])
    else:
        (x1, y1, w1, h1) = cv2.boundingRect(tablica)


    return((x1, y1, x1+w1, y1+h1))

def endtoend(iteracija, granica):
    #brojac odradjenih slika
    brojac = 0
    iou = []
    tpr = []
    fpr = []
    ukupno = len(os.listdir(os.path.join(link, '')))/2
    print_progress_bar(0, 100, prefix=(
        "\t"+link+"\t"+str(iteracija)+"\t"), suffix=("\t∞\t"+str(granica)+"\t"))
    
    #Prolazak kroz svaki .txt fajl u folderu (pogledati format .txt fajl-a)
    for filename in os.listdir(os.path.join(link, '')):
        if filename.endswith(".txt"):
            #Citanje podataka iz fajla i ucitavanje slike
            f = open(os.path.join(link, f"{filename}"), "r")
            txt = f.read().split('\t')
            fajl = txt[0]
            img = cv2.imread(os.path.join(link, f"{fajl}"))
            name = fajl.split('.')[0]
            sx = int(txt[1])
            sy = int(txt[2])
            #Pazi na bazu koju koristis jer od toga zavisi da li je ovo
            #w i h ili koordinate donjeg desnog temena pravougaonika
            w = int(txt[3])
            h = int(txt[4])
            #isto sto i gore napisano
            ex = sx + w
            ey = sy + h
            #detektcija tablice
            (sx1, sy1, ex1, ey1) = tablica(img, name, granica)
            brojac += 1
            #definisanje pravougaonika i povrsina (Rectangle->namedTuple)
            detektovano = Rectangle(sx1, sy1, ex1, ey1)
            baza = Rectangle(sx, sy, ex, ey)
            height, width = img.shape[:2]
            povrsBaza = (ex-sx)*(ey-sy)
            povrsDetekt = (ex1-sx1)*(ey1-sy1)
            povrsSlika = width*height
            #izracunavanje Karakteristika
            TP = area(baza, detektovano)
            FN = povrsBaza-TP
            FP = povrsDetekt-TP
            TN = povrsSlika-TP-FN-FP
            TPR = TP/(TP+FN)*100
            FPR = FP/(FP+TN)*100
            IOU = TP/(FP+TP+FN)*100
            #dodavanje na niz karakteristika (globalna varijabla)
            iou.append(IOU)
            tpr.append(TPR)
            fpr.append(FPR)
            #izracunavanje preostalog vremena za trenutnu iteraciju
            dosad = time.time()-start
            prosecnovreme = dosad/len(iou)
            preostalovreme = (ukupno-brojac) * prosecnovreme
            if preostalovreme >= 60:
                preostalovreme = str(int(preostalovreme/60)) + " min"
            else:
                preostalovreme = str(int(preostalovreme))+" s"
            #ispisivanje progress bar-a sa svim potrebnim podacima, link ,iteracija, preostalo vreme i granica
            print_progress_bar(brojac, ukupno, prefix=(
                "\t"+link+"\t"+str(iteracija)+"\t"), suffix=("\t"+preostalovreme+"\t"+str(granica)+"\t"))
    return(iou, tpr, fpr)
#ucitavanje fajla za rezultate
text_file = open("Rezultati.txt", "w")
IOU = []
TPR = []
FPR = []
#link do slika i upisivanje istog u fajl za rezultate
link = os.path.join('..', 'benchmarks','endtoend','eu')
text_file.write(
    link+'\t'+str(int(len(os.listdir(os.path.join(link, '')))/2))+'\n')
text_file.flush()

#definisanje broja iteracija ==> gustina prelazenja preko intervala [0,1]
first = 0
last = 100
#promena granice za jednu iteraciju
delta=1/(last-first)
#merenje vremena od pocetka izvrsavanja programa (izostavljaci inicijalni period, za preciznije merenje samog algoritma)
program_start = time.time()
for iteracija in range(first, last+1, 1):
    #Nivo sigurnosti detekcije OCR-a
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
plt.scatter(x,TPR)
fig.suptitle("TPR")
plt.ylim(0, 100)
plt.xlabel('Nivo sigurnosti detekcije OCR-a')
plt.ylabel('TPR [%]')
fig.savefig(os.path.join('..', 'Rezultati', 'TPR.svg'))

fig = plt.figure()
plt.plot(x, FPR)
plt.scatter(x,FPR)
fig.suptitle("FPR")
plt.ylim(0, 100)
plt.xlabel('Nivo sigurnosti detekcije OCR-a')
plt.ylabel('FPR [%]')
fig.savefig(os.path.join('..', 'Rezultati', 'FPR.svg'))

fig = plt.figure()
plt.plot(FPR, TPR)
plt.scatter(FPR,TPR)
fig.suptitle("ROC")
plt.ylim(0, 100)
plt.xlabel('FPR [%]')
plt.ylabel('TPR [%]')
fig.savefig(os.path.join('..', 'Rezultati', 'ROC.svg'))