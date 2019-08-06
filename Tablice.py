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
    cetiritacke = [[a.xmin, a.ymin], [a.xmax, a.ymax], [b.xmin, b.ymin], [b.xmax, b.ymax]]
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

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total: 
        print('\n')    
    

def merge (textbox1, textbox2):
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


    

def tablica(img,name):
    orig=img.copy()
    #text_file.write(str(broj)+"\n") 

    img=orig.copy()
    prikaz=orig.copy()
    textboxes=[]
    granica=1
    while len(textboxes)<1:
        textboxes=text_detection.text_detection(img,granica)
        '''for txt in textboxes:
            if abs(txt[0] - txt[2]) > 500 or abs(txt[1] - txt[3]) > 300 or abs(txt[0] - txt[2]) / abs(txt[1] - txt[3]) < 1.5:
                textboxes.remove(txt)'''
        granica-=0.05
    if len(textboxes) > 1:
        i = 0
        while i < len(textboxes):
            j = i + 1
            while j < len(textboxes):
                if (abs(textboxes[i][2] - textboxes[j][0]) < 20 or abs(textboxes[i][0] - textboxes[j][2]) < 20) or (abs(textboxes[i][3] - textboxes[j][1]) < 10 or abs(textboxes[i][1] - textboxes[j][3]) < 10):
                    textboxes.insert(i + 1, merge(textboxes[i], textboxes[j]))
                    textboxes.pop(i + 1)
                    textboxes.pop(j)
                j+=1
            i+=1

    #img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)
    add = cv2.add(value, topHat)
    subtract = cv2.subtract(add, blackHat)
    blur = cv2.GaussianBlur(subtract, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    
    CannyThresh=cv2.Canny(thresh,0,200,10)

    """
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gauss=cv2.GaussianBlur(img,(5,5),0)
    gaussgray=cv2.GaussianBlur(gray,(5,5),0)
    gaussCanny=cv2.Canny(gauss,100,200,10)
    gaussCannygray=cv2.Canny(gaussgray,0,200,10)
    overlap=cv2.bitwise_and(gaussCanny,gaussCannygray)
    """

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
    (contours,__)=cv.findContours( CannyThresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);
    (x1,y1,w1,h1)=(0,0,0,0)

    height,width = img.shape[:2]
    blank_image = np.zeros((height,width,3), np.uint8)
    blank_image=cv2.cvtColor(blank_image,cv2.COLOR_BGR2GRAY)
    cv2.drawContours(blank_image,contours,-1,(255,255,255),1)
    blank_image=cv2.cvtColor(blank_image,cv2.COLOR_GRAY2RGB)
    blanksave=Image.fromarray(blank_image)
    blanksave.save("../Rezultati/{0}_ivice.jpg".format(name))

    dobrekonture=[]
    mogucetablice=[]

    cv.drawContours(prikaz, contours, -1, (0,0,255), 1)

    for (sx,sy,ex,ey) in textboxes:
        cv.rectangle(prikaz, (sx,sy), (ex,ey), (255,255,0), 2)
        
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        overlay = prikaz.copy()

        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 1)
        alpha = 0.7 
        prikaz = cv2.addWeighted(overlay, alpha, prikaz, 1 - alpha, 0)

        #cv2.rectangle(prikaz, (x,y), (x+w,y+h), (0,255,0), 2)
        if 2<=(w/h)<=1000 and 1000>w>50 and 1000>h>0:
            tester=orig[y:y+h,x:x+w]
            tester=cv.cvtColor(tester, cv.COLOR_BGR2GRAY)
            (meanbright1,__,__,__)=cv.mean(tester)
            #cv2.rectangle(prikaz, (x,y), (x+w,y+h), (0,255,0), 2)


            if 0<=(w/h)<=1000:
                if meanbright1>=0:
                    cv2.rectangle(prikaz, (x,y), (x+w,y+h), (255,0,0), 2)
                    dobrekonture.append(contour)
                else:
                    cv2.rectangle(prikaz, (x,y), (x+w,y+h), (0,255,255), 2)
                    
        
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    imgbw=cv.cvtColor(orig,cv.COLOR_BGR2GRAY)
        
        
    for contour in dobrekonture:
        A=0
        score=0
        (x,y,w,h) = cv2.boundingRect(contour)
        #index=dobrekonture.index(contour)
        #cv2.putText(prikaz,str(index),(x,y), font, 1,(0,0,0),2,cv2.LINE_AA)
        tester=imgbw[y:y+h,x:x+w]
        bound=Rectangle(x,y,x+w,y+h)
        boundarea=w*h
        histr = cv2.calcHist([tester],[0],None,[256],[0,256])
        beli=sum(histr[128:255])
        crni=sum(histr[0:127])
        #odnos=beli/crni
        """
        if crni>0:
            odnos=beli/crni
        else:
            odnos=0
        """
        for (sx,sy,ex,ey) in textboxes:
            r=Rectangle(sx,sy,ex,ey)
            #print("Area:", r, bound, area(r,bound))
            if area(r,bound)/boundarea*100 + area(r,bound)/((ex-sx)*(ey-sy)) * 70 > A:
                A = area(r,bound)/boundarea*100 + area(r,bound)/((ex-sx)*(ey-sy)) * 70
            #print(area(r,bound)/boundarea)
            
        #if odnos>3:
        #    odnos=0
        #odstup=abs(4-(w/h))
        #print(A)    
        score=A 
        #+ odnos*10
        #50*A+
        #+w*h/30000+odstup/50
            
        #+min((w-40)/5+(h-10)/5,50)+min(w*h/50,50)+min(50/odstup,50)
        cv2.putText(prikaz,str(score),(x,y), font, 0.5,(0,0,0),2,cv2.LINE_AA,)
        #if(odnos>=0.1):
        mogucetablice.append((score,contour))
                
        
        
    prikazslika=Image.fromarray(prikaz)
    prikazslika.save("../Rezultati/{0}_detekcija.jpg".format(name))
    if len(mogucetablice)>0:
        maxscore,tablica=mogucetablice[0]
        for (score,contour) in mogucetablice:
            #if(score!=0):
            if 1==1:
                (x,y,w,h) = cv2.boundingRect(contour)
                tester=orig[y:y+h,x:x+w]
                if(score>maxscore):
                    maxscore,tablica=score,contour
            
        
    
    
    orig = cv.cvtColor(orig, cv.COLOR_BGR2RGB)

    try:
        (x1,y1,w1,h1) = cv2.boundingRect(tablica)
        orig=orig[y1:y1+h1,x1:x1+w1]
        raise
    except:
        cv2.putText(orig,"OVO JE LOS PROGRAM!!!!",(10,300), font, 2,(255,0,0),10,cv2.LINE_AA)
    
    text=pytesseract.image_to_string(orig)
    #print("Pytesseract: ", text)
    """
    plt.figure()
    plt.imshow(prikaz)
    plt.figure()
    plt.imshow(orig)
    plt.show()
    """
    #text_file.write(text+"\n") 
    orig=cv.cvtColor(orig,cv.COLOR_BGR2RGB)
    origslika=Image.fromarray(orig)
    origslika.save("../Rezultati/{0}_tablica.jpg".format(name))
    
    return((x1,y1,x1+w1,y1+h1),granica,text)

text_file = open("zuti_pravougaonici_Odnos.txt", "w")
metrike=[]
link="../benchmarks/endtoend/eu/"

def endtoend():
    brojac=0
    ukupno=len(os.listdir(link))/2
    printProgressBar (0, 100, suffix="\t 0%")
    for filename in os.listdir(link):
        if filename.endswith(".txt"):

            f = open(link+"{0}".format(filename), "r")
            txt=f.read().split('\t')
            fajl=txt[0]
            img=cv2.imread(link+"{0}".format(fajl))
            name=fajl.split('.')[0]
            sx=int(txt[1])
            sy=int(txt[2])
            w=int(txt[3])
            h=int(txt[4])
            ex=sx+w
            ey=sy+h
            (sx1,sy1,ex1,ey1),granica,text=tablica(img,name)
            
            detektovano=Rectangle(sx1,sy1,ex1,ey1)
            baza=Rectangle(sx,sy,ex,ey)
            presek=area(baza,detektovano)
            unija=(ex1-sx1)*(ey1-sy1)+(ex-sx)*(ey-sy)-presek
            iou=presek/unija*100
            #print(name)
            text_file.write(name+" "+str(iou)+" % ("+str(granica)+") ["+text+"]\n")
            metrike.append(iou)

            currmetrika=round(sum(metrike)/len(metrike),2)
            brojac+=1
            printProgressBar (brojac, ukupno-1, suffix=("\t"+str(currmetrika)+" %"))
            


"""def grci():
    for filename in os.listdir('Slike'):
        img = cv2.imread(filename)
"""



start=time.time()
endtoend()
vreme=time.time()-start   

text_file.write("\n")
metrika=sum(metrike)/len(metrike)
brojac=0
suma=0
prosecnovreme=vreme/len(metrike)
for i in metrike:
    if i>50:
        suma+=i
        brojac+=1
m1=suma/brojac

plt.hist(metrike,bins='auto')
plt.savefig('histogram_zuti_odnos.jpg')

print("\r Ukupno: "+str(metrika)+" % \r")
text_file.write("Ukupno: "+str(metrika)+" % ("+str(len(metrike))+")\n")
text_file.write("Vece od 50%: "+str(m1)+" % ("+str(brojac)+")\n")
text_file.write("Prosecno vreme po slici: "+str(prosecnovreme)+" s ("+str(vreme/60)+" min)\n")
text_file.close()