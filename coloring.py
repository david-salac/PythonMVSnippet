import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cv2
import sys
from copy import deepcopy

#---------Ukol 01---------
img = cv2.imread('cv09_bunkyB.bmp', 2);

kernel = np.zeros((2,3), np.uint8);
kernel[0,0] = 0;
kernel[0,1] = 1;
kernel[0,2] = 0;
kernel[1,0] = 1;
kernel[1,1] = 1;
kernel[1,2] = 1;

obr = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel);
obr = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel);

plt.figure('Ukol 1 - Otevreni a uzavreni na sum');
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray');
plt.title('Puvodni se sumem');
plt.subplot(1,2,2)
plt.imshow(obr, cmap='gray');
plt.title('Novy bez sumu');
plt.show();

#---------Ukol 02---------
img = cv2.imread('cv09_rice.bmp', 2);

tophatKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, tophatKernel)

plt.figure('Ukol 2 - operace tophat');
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray');
plt.title('Puvodni obrazek');
plt.subplot(1,2,2)
plt.imshow(tophat, cmap='gray');
plt.title('Po provedeni operace tophat');
plt.show();


#---------Ukol 3---------
#Funkce pro segmentaci obrazku obr
def segmentaceHistogram(obr, prah):
    newObr = np.zeros((obr.shape[0],obr.shape[1]), dtype=np.int);
    for y in range(0,obr.shape[0]):
        for x in range(0, obr.shape[1]):
            if (int(obr[y,x]) < prah ): #Barevna slozka G*255/(R+G+B)
                newObr[y,x] = 0;
            else:
                newObr[y,x] = 1;
    return newObr;


def segmentaceObrazu(obr, prah):
    hist, temp = np.histogram(obr[:], 256, (0, 256))
    '''plt.figure('Histogram obrazku');
    plt.plot(hist);
    plt.show();'''
    #Vstup pro prah
    #prah = 50;#input("Zadejte hodnotu prahu: ");
    #prah = int(prah);
    return hist, segmentaceHistogram(obr, prah);

histogram, segmentovany = segmentaceObrazu(tophat, 50);
histogram=histogram/max(histogram);

plt.figure('Ukol 3 - prahovani');
plt.subplot(1,3,1)
plt.imshow(tophat, cmap='gray');
plt.title('Puvodni obrazek');
plt.subplot(1,3,2)
plt.imshow(segmentovany, cmap='gray');
plt.title('Po prahovani');
plt.subplot(1,3,3)
plt.plot(histogram);
plt.title('Histogram');
plt.show();

#---------Ukol 4 a 5---------
#Funkce vraci hodnoty okoli obrazku
def sousedniHodnoty(obr, x, y):
    hodnoty = [];
    if obr[y, x-1] not in hodnoty:
        hodnoty.append(obr[y, x-1]);
    if obr[y, x] not in hodnoty:
        hodnoty.append(obr[y, x]);
    if obr[y-1, x - 1] not in hodnoty:
        hodnoty.append(obr[y-1, x - 1]);
    if obr[y-1, x] not in hodnoty:
        hodnoty.append(obr[y-1, x]);
    if obr[y-1, x + 1] not in hodnoty:
        hodnoty.append(obr[y-1, x + 1]);
    return sorted(hodnoty);

#Vraci barvy z okoli, tj. hodnoty mimo 0 a 1
def sousedniBarvy(seznamHodnot, aktualniBarva):
    seznamBarev = [];
    for i in range(0, len(seznamHodnot)):
        if seznamHodnot[i] > 1 and seznamHodnot[i] != aktualniBarva:
            seznamBarev.append(seznamHodnot[i]);
    return seznamBarev;

#Vytvori tabulku barevnych identit, rekurzivne hleda identity
def vratIdentity(pole, hledam):
    identity = [];
    for i in range (0, len(pole)):
        if pole[i][0] == hledam:
            pridat = pole[i][1];
            identity.append(pridat);
            pole[i] = [0,0];
            identity.extend(vratIdentity(pole, pridat))
        if pole[i][1] == hledam:
            pridat = pole[i][0];
            identity.append(pridat);
            pole[i] = [0,0];
            identity.extend(vratIdentity(pole, pridat))
    return identity;


#Funkce barvi obrazek
def barveniOblasti(obr):
    obr = np.mod(obr, 2);
    
    barevneIdentity = [];
    #barvyVObraze = [];

    citacOdstinu = 2;
    vybranyOdstin = 0;
    
    for y in range(1, obr.shape[0] - 1):
        for x in range(1, obr.shape[1] -1):
            #Barvim pouze pixely s hodnotou 1
            if obr[y,x] > 0:
                #Vybere vsechny hodnoty ze 4 okoli
                okoli = sousedniHodnoty(obr,x,y);
                if citacOdstinu not in okoli: #Pokud neni zvolena hodnota citace odstinu v okoli, vytvori se nova hodnota
                    citacOdstinu += 1;
                obr[y,x] = citacOdstinu; #Priradi se nova/stara hodnota z citace odstinu do aktualniho pixelu
                
                #Dale se zjistuje, jestli nejsou v okoli nejake jine barvy, pokud ano, priradi se pixelu jejich hodnota
                barvy = sousedniBarvy(okoli, citacOdstinu); #Vybere vsechny barvy z okoli (vyjma 0 a 1)
                #Zde se resetuje hodnota pro barveni odstinem z okoli, pokud zadny takovy neni
                if vybranyOdstin not in barvy:
                    vybranyOdstin = 0; #Resetuje hodnotu pro vyber odstinu na nulu
                #Pokud se v okoli nachazi nejaka barva, bude pouzita k barveni (je li jich vice, vezme se minimum)
                if len(barvy) > 0 and vybranyOdstin == 0:
                    vybranyOdstin = barvy[0]; 
                
                if vybranyOdstin != 0: #Zde se barvi odstinem z okoli
                    obr[y,x] = vybranyOdstin;
                
                #Zde se resi barevne identity
                noveBarvy = sousedniBarvy(okoli, obr[y,x]); #Vybere vsechny barvy z okoli (vyjma 0 a 1)
                
                for k in range(0, len(noveBarvy)):
                    if noveBarvy[k] > obr[y,x]:
                        barevneIdentity.append([obr[y,x], noveBarvy[k]]);
                    else:
                        barevneIdentity.append([noveBarvy[k], obr[y,x]]);
                
    
    #Krok 2: Odstraneni duplicit z barevnych identit
    identity = [];
    for j in range(0, len(barevneIdentity)):
        if barevneIdentity[j] not in identity:
            identity.append(barevneIdentity[j]);

    #Vytrvoreni tabulky identit
    noveBarvy = []
    for i in range(0, len(identity)):
        if identity[i][0] != 0:
            identita = [identity[i][0]];
            identita.extend(vratIdentity(identity, identity[i][0]));
            noveBarvy.append(identita);
    #KROK 3: Barevna paleta a obarveni
    paleta = [];
    #Zaroven vypocet teziste
    tezisteX = [];
    tezisteY = [];
    tezisteSuma = [];
    
    for x in range(0, obr.shape[1]):
        for y in range(0, obr.shape[0]):
            barva = obr[y,x];
            for i in range(0, len(noveBarvy)):
                if obr[y,x] in noveBarvy[i]:
                    obr[y,x] = noveBarvy[i][0];
            if obr[y,x] not in paleta:
                paleta.append(obr[y,x]);
                tezisteX.append(0);
                tezisteY.append(0);
                tezisteSuma.append(0);  
            
            obr[y,x] = paleta.index(obr[y,x]);
            tezisteX[obr[y,x]] += x;
            tezisteY[obr[y,x]] += y;
            tezisteSuma[obr[y,x]] += 1;
        
    #Vypocet teziste
    for i in range(0, len(paleta)):
        tezisteX[i] /= tezisteSuma[i];
        tezisteY[i] /= tezisteSuma[i];
    
    return obr, tezisteSuma, tezisteX, tezisteY;

#Vlastni barveni
nwObr, tezisteSum, tezX, tezY = barveniOblasti(segmentovany);

tezisteX = [];
tezisteY = [];
pocet = 0;

for i in range(2, len(tezisteSum)):
    if (tezisteSum[i] >= 90):
        tezisteX.append(int(tezX[i]));
        tezisteY.append(int(tezY[i]));
        pocet += 1;

plt.figure('Ukol 4 a 5 - hledani zrnicek');
plt.subplot(1,2,1)
plt.imshow(nwObr, cmap='inferno');
plt.title('Obarveny obrazek');
plt.subplot(1,2,2)
plt.imshow(img, cmap='gray');
plt.plot(tezisteX, tezisteY, 'ro');
plt.title('Teziste objektu, pocet = ' + str(pocet));
plt.show();

