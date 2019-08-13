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

CRTAJ = False
if CRTAJ:
    import pytesseract


def obrada_blackhat(img):
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(img, (5, 5), 0)
    gaussgray = cv2.GaussianBlur(gray, (5, 5), 0)
    gaussCanny = cv2.Canny(gauss, 100, 200, 10)
    gaussCannygray = cv2.Canny(gaussgray, 0, 200, 10)
    overlap = cv2.bitwise_and(gaussCanny, gaussCannygray)
    return(overlap)


def obrada_canny_stari(img):
    img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    meanbright, _, _, _ = cv.mean(img)
    img = cv.blur(img, (5, 5))
    thres1, img1 = cv2.threshold(img, meanbright, 255, 0)
    img1 = cv.Canny(img, 100, 200, 10)
    img1 = cv2.dilate(img1, (3, 3), 1)
    img1 = cv2.erode(img1, (3, 3), 1)
    img1 = cv2.dilate(img1, (3, 3), 10)
    return(img1)


def area(a: namedtuple, b: namedtuple) -> int:
    """
    Funkcija koja racuna povrsinu preseka dva pravougaonika
    :param a:
    :param b:
    :return:
    """
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


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print('\n')


def merge(textbox1: list, textbox2: list) -> tuple:
    """

    :param textbox1:
    :param textbox2:
    :return:
    """
    textbox3: list = []
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
    """

    :param img:
    :param sx:
    :param sy:
    :param ex:
    :param ey:
    :return:
    """
    img = img[sy:ey, sx:ex]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    avg_hue = np.average(hue)/255*360
    avg_sat = np.average(saturation)/255*100
    avg_val = np.average(value)/255*100

    return avg_hue, avg_sat, avg_val


def tablica(img: Image, name):
    """

    :param img:
    :param name:
    :return:
    """
    # Flag za proveru postojanja preseka
    flag_presek = 1
    orig = img.copy()
    kopija = orig.copy()

    if CRTAJ:
        prikaz = orig.copy()

    textboxes = []
    # Granica sigurnosti OCR
    granica = 1
    while len(textboxes) < 1:
        textboxes = text_detection.text_detection(img, granica)
        granica -= 0.05

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

    # ALGORITMI OBRADE
    processing_out = obrada_blackhat(img)

    (contours, __) = cv.findContours(
        processing_out, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )

    if CRTAJ:
        height, width = img.shape[:2]
        blank_image = np.zeros((height, width, 3), np.uint8)
        blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
        cv2.drawContours(blank_image, contours, -1, (255, 255, 255), 1)
        blank_image = cv2.cvtColor(blank_image, cv2.COLOR_GRAY2RGB)
        blanksave = Image.fromarray(blank_image)
        img_name = os.path.join('..', 'RezultatiBrazil', f"{name}_ivice.jpg")
        blanksave.save(img_name)
        cv.drawContours(prikaz, contours, -1, (0, 0, 255), 1)
        for (start_x, start_y, end_x, end_y) in textboxes:
            cv.rectangle(
                prikaz, (start_x, start_y), (end_x, end_y), (255, 255, 0), 2
            )

    dobre_konture = []
    moguce_tablice = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if CRTAJ:
            overlay = prikaz.copy()
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 1)
            alpha = 0.7
            prikaz = cv2.addWeighted(overlay, alpha, prikaz, 1 - alpha, 0)

        if 1.5 <= (w/h) <= 10 and 1000 > w > 100 and 1000 > h > 5:
            if CRTAJ:
                cv2.rectangle(prikaz, (x, y), (x+w, y+h), (255, 0, 0), 2)
            dobre_konture.append(contour)

    font = cv2.FONT_HERSHEY_SIMPLEX

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

        if CRTAJ:
            cv2.putText(prikaz, str(round(score, 0)), (x, y),
                        font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            img_name = os.path.join(
                '..', 'RezultatiBrazil', f"{name}_detekcija.jpg")
            prikaz_slika = Image.fromarray(prikaz)
            prikaz_slika.save(img_name)

    if len(moguce_tablice) > 0:
        maxscore, tablica = moguce_tablice[0]
        for (score, contour) in moguce_tablice:
            (x, y, w, h) = cv2.boundingRect(contour)
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

    if flag_presek == 0:
        (x1, y1, w1, h1) = cv2.boundingRect(tablica)
        (_, sat1, _) = histogrami(img, x1, y1, x1 + w1, y1 + h1)
        (_, sat2, _) = histogrami(
            img, textboxes[pozicija2][0], textboxes[pozicija2][1], textboxes[pozicija2][2], textboxes[pozicija2][3])

        if sat2 >= sat1:
            (x1, y1, w1, h1) = tuple(textboxes[pozicija2])

    else:
        (x1, y1, w1, h1) = cv2.boundingRect(tablica)

    if CRTAJ:
        text = pytesseract.image_to_string(orig)
    else:
        text = ""

    if CRTAJ:
        orig = orig[y1:y1+h1, x1:x1+w1]
        orig = cv.cvtColor(orig, cv.COLOR_BGR2RGB)
        img_name = os.path.join('..', 'RezultatiBrazil', f"{name}_tablica.jpg")
        origslika = Image.fromarray(prikaz)
        origslika.save(img_name)

    return((x1, y1, x1+w1, y1+h1), granica, text)


text_file = open("Rezultati.txt", "w")
#text_file = open("Rezultati.txt", "a")
link = os.path.join('..', 'benchmarks', 'endtoend', 'eu')
text_file.write(link+'\n')


def endtoend():
    brojac = 0
    iou = []
    tpr = []
    fpr = []
    ukupno = len(os.listdir(os.path.join(link, '')))/2
    print_progress_bar(0, 100, prefix=("\t" + os.path.join(link, '') + "\t"))
    text_file.flush()

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
            (sx1, sy1, ex1, ey1), granica, text = tablica(img, name)
            (avgHue, avgSat, avgVal) = histogrami(img, sx1, sy1, ex1, ey1)
            brojac += 1
            detektovano = Rectangle(sx1, sy1, ex1, ey1)
            baza = Rectangle(sx, sy, ex, ey)
            presek = area(baza, detektovano)
            height, width = img.shape[:2]
            povrsBaza = (ex-sx)*(ey-sy)
            povrsDetekt = (ex1-sx1)*(ey1-sy1)
            povrsSlika = width*height
            TP = area(baza, detektovano)
            FN = povrsBaza-TP
            FP = povrsDetekt-TP
            TN = povrsSlika-TP-FN-FP
            TPR = TP/(TP+FN)*100
            FPR = FP/(FP+TN)*100
            IOU = TP/(FP+TP+FN)*100

            iou.append(IOU)
            tpr.append(TPR)
            fpr.append(FPR)

            text_file.write(str(len(iou))+"\t"+name+"\t"+str(round(IOU, 2))+"\t"+str(round(TPR, 2))+"\t" + str(round(
                FPR, 2))+"\t"+str(round(granica, 2))+"\t"+str(round(avgSat, 1))+"\n")
            text_file.flush()

            dosad = time.time()-start
            prosecnovreme = dosad/len(iou)
            preostalovreme = (ukupno-brojac) * prosecnovreme
            if preostalovreme >= 60:
                preostalovreme = str(round(preostalovreme/60, 1)) + " min"
            else:
                preostalovreme = str(round(preostalovreme, 1))+" s"
            print_progress_bar(brojac, ukupno, prefix=(
                "\t"+link+"\t"), suffix=("\t"+preostalovreme+"\t"))
    return(iou, tpr, fpr)


start = time.time()
iou, tpr, fpr = endtoend()
vreme = time.time()-start

_iou = sum(iou)/len(iou)
_tpr = sum(tpr)/len(tpr)
_fpr = sum(fpr)/len(fpr)

text_file.write("\n")
prosecnovreme = vreme/len(iou)

plt.figure()
plt.hist(iou, bins='auto')
plt.savefig('Rezultati_histogram.jpg')

text_file.write(str(round(_iou, 2))+"\t"+str(round(_tpr, 2))+"\t" + str(round(
    _fpr, 2))+"\t"+str(len(iou))+"\t\n")
text_file.write(+str(round(prosecnovreme, 2)) +
                " s ("+str(round(vreme/60, 2))+" min)\n\n")
text_file.close()
