# Main.py

from importlib.resources import path
from unicodedata import name
import cv2
import numpy as np
import os
import time
import serial
import xlsxwriter

import DetectChars
import DetectPlates
import PossiblePlate

# variabel tingkat modul ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False
parent_dir = "D:/CODING PEAN29/Deteksi Plat/Proses/"
platDetected= []
global folderPath
ser = serial.Serial('COM3', 9800, timeout=1)
time.sleep(1)


###################################################################################################
def main(image):
    folder = str(time.strftime("%H%M", time.localtime()))
    folderPath = folder
    path = parent_dir + folderPath
    checkFolder = os.path.exists(path)
    if checkFolder == False:
        os.mkdir(path)
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # mencoba pelatihan KNN
    if blnKNNTrainingSuccessful == False:                               # jika pelatihan KNN tidak berhasil
        print("\nerror: KNN traning was not successful\n")  # tampilkan pesan kesalahan
        return                                                          # dan keluar dari program
    # berakhir jika
    # membuka file gambar
    imgSource  = image
    # merubah gambar ke warna negatif 
    reverseImg = cv2.bitwise_not(imgSource)
    imgOriginalScene  = reverseImg

    if imgSource is None:                            # jika gambar tidak berhasil dibaca
        print("\nerror: image not read from file \n\n")  # cetak pesan kesalahan ke std out
        os.system("pause")                                  # jeda sehingga pengguna dapat melihat pesan kesalahan
        return                                              # dan keluar dari program
    # berakhir jika

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # mendeteksi plat

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # mendeteksi karakter di plat

    #cv2.imshow("imgOriginalScene", imgAdegan Asli)            # tampilkan gambar pemandangan

    if len(listOfPossiblePlates) == 0:                          # jika tidak ada plat yang ditemukan

        print("\nno license plates were detected\n")  # beri tahu pengguna tidak ada plat yang ditemukan
    else:                                                   # kalau tidak
                # jika kita masuk ke sini daftar kemungkinan plat memiliki setidaknya satu plat



                # urutkan daftar plat yang mungkin dalam urutan DESCENDING (jumlah karakter terbanyak ke jumlah karakter paling sedikit)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                # misalkan plat dengan karakter yang paling dikenal (plat pertama yang diurutkan berdasarkan urutan menurun panjang string) adalah plat yang sebenarnya
        licPlate = listOfPossiblePlates[0]

        #cv2.imshow("imgPlate", licPlate.imgPlate)           # tunjukkan potongan piring dan ambang batas plat
        #cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                     # jika tidak ada karakter yang ditemukan di plat
            print("\nno characters were detected\n\n")  # tunjukkan pesan
            ser.write(b'L')
            time.sleep(0.1)
            return                                          # dan keluar dari program
        else:
            ser.write(b'H')
            time.sleep(0.1)  
        # berakhir jika

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # dpersegi panjang merah mentah di sekitar plat

        print("\nlicense plate read from image = " + licPlate.strChars + "\n")  # tulis teks plat nomor ke std out
        print("----------------------------------------")

        platDetected.append(licPlate.strChars)
        writeLicensePlateCharsOnImage(imgSource, licPlate)           # tulis teks plat nomor pada gambar

        cv2.imshow("imgOriginalScene", imgSource)                # tampilkan ulang gambar adegan
        name = 'Hasil/Plat(' + licPlate.strChars +').jpg'
        cv2.imwrite(name, imgSource)           # tulis gambar ke file

    # akhiri jika lain

    #cv2.waitKey(0)					# tahan windows terbuka sampai pengguna menekan tombol

    return
# end main

###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # dapatkan 4 simpul dari persegi yang diputar

    #cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # menggambar 4 garis merah
    #cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    #cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    #cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             # ini akan menjadi pusat area tempat teks akan ditulis
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          # ini akan menjadi kiri bawah area tempat teks akan ditulis
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # pilih font jane polos
    fltFontScale = float(plateHeight) / 60.0                    # skala font dasar pada ketinggian area pelat
    intFontThickness = int(round(fltFontScale * 2))           # ketebalan font dasar pada skala font



    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # panggil getTextSize

            # membongkar rotated rect ke titik tengah, lebar dan tinggi, dan sudut
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              # mpastikan pusat adalah bilangan bulat
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         # lokasi horizontal area teks sama dengan plat

    if intPlateCenterY < (sceneHeight * 0.75):                                                  # jika plat nomor ada di 3/4 bagian atas gambar
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # tulis karakter di bawah plat
    else:                                                                                       # lain jika plat nomor ada di 1/4 bagian bawah gambar
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # tulis karakter di atas plat
    # end if

    textSizeWidth, textSizeHeight = textSize                # membongkar lebar dan tinggi ukuran teks

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           # hitung asal kiri bawah area teks
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          # berdasarkan pusat area teks, lebar, dan tinggi

            # tulis teks pada gambar
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
# fungsi akhir

###################################################################################################


















