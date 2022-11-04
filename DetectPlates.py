# DetectPlates.py

import os
import cv2
import numpy as np
import math
import Main
import random
import time

import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar

# variabel tingkat modul ##########################################################################
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.1

folderName = "Proses/" + str(time.strftime("%H%M", time.localtime()))
###################################################################################################
def detectPlatesInScene(imgOriginalScene):
    listOfPossiblePlates = []                   # ini akan menjadi nilai pengembalian

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    if Main.showSteps == True: # menunjukkan langkah-langkah #######################################################
        cv2.imshow("0", imgOriginalScene)
    name0 = folderName + '/0.jpg'
    cv2.imwrite( name0, imgOriginalScene)
    # berakhir jika # menunjukkan langkah-langkah #########################################################################

    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(imgOriginalScene)         # praproses untuk mendapatkan gambar skala abu-abu dan biner

    if Main.showSteps == True: # menunjukkan langkah-langkah #######################################################
        cv2.imshow("1a", imgGrayscaleScene)
        cv2.imshow("1b", imgThreshScene)
    name1a = folderName + '/1a.jpg'
    cv2.imwrite( name1a, imgGrayscaleScene)
    name1b = folderName + '/1b.jpg'
    cv2.imwrite( name1b, imgThreshScene)
    # berakhir jika # menunjukkan langkah-langkah #########################################################################

            # temukan semua karakter yang mungkin dalam adegan,
            # fungsi ini pertama-tama menemukan semua kontur, kemudian hanya menyertakan kontur yang dapat berupa karakter (belum dibandingkan dengan karakter lain)
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)

    if Main.showSteps == True: # menunjukkan langkah-langkah #######################################################
        print("step 2 - len(listOfPossibleCharsInScene) = " + str(
            len(listOfPossibleCharsInScene)))  # 131 with MCLRNF1 image

        imgContours = np.zeros((height, width, 3), np.uint8)

        contours = []

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)
        # end for

        cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
        cv2.imshow("2b", imgContours)
    # end if # menunjukkan langkah-langkah #########################################################################

            # diberikan daftar semua karakter yang mungkin, temukan grup karakter yang cocok
            # pada langkah selanjutnya setiap kelompok karakter yang cocok akan berusaha dikenali sebagai pelat
    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    if Main.showSteps == True: # show steps #######################################################
        print("step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(
            len(listOfListsOfMatchingCharsInScene)))  # 13 with MCLRNF1 image

        imgContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
        # end for

        cv2.imshow("3", imgContours)
    # end if # menunjukkan langkah-langkah #########################################################################

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:                   # untuk setiap grup karakter yang cocok
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)         # mencoba untuk mengekstrak plat

        if possiblePlate.imgPlate is not None:                          # jika plat ditemukan
            listOfPossiblePlates.append(possiblePlate)                  # tambahkan ke daftar kemungkinan plat
        # end if
    # end for

    print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")  # 13 with MCLRNF1 image

    if Main.showSteps == True: # menunjukkan langkah-langkah #######################################################
        print("\n")
        cv2.imshow("4a", imgContours)
        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), Main.SCALAR_RED, 2)

            cv2.imshow("4a", imgContours)

            print("possible plate " + str(i) + ", click on any image and press a key to continue . . .")

            cv2.imshow("4b", listOfPossiblePlates[i].imgPlate)
            name4b = folderName + '/4b.jpg'
            cv2.imwrite( name4b, listOfPossiblePlates[i].imgPlate)
            cv2.waitKey(0)
        # berakhir untuk

        print("\nplate detection complete, click on any image and press a key to begin char recognition . . .\n")
        cv2.waitKey(0)
    # berakhir jika # show steps #########################################################################

    return listOfPossiblePlates
# fungsi akhir

###################################################################################################
def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []                # ini akan menjadi nilai pengembalian

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # find all contours

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):                       # untuk setiap kontur

        if Main.showSteps == True: # show steps ###################################################
            cv2.drawContours(imgContours, contours, i, Main.SCALAR_WHITE)
        # end if # show steps #####################################################################

        possibleChar = PossibleChar.PossibleChar(contours[i])

        if DetectChars.checkIfPossibleChar(possibleChar):                   # jika kontur adalah karakter yang memungkinkan, perhatikan ini tidak dibandingkan dengan karakter lain (belum) . . .
            intCountOfPossibleChars = intCountOfPossibleChars + 1           # jumlah kenaikan kemungkinan karakter
            listOfPossibleChars.append(possibleChar)                        # dan tambahkan ke daftar kemungkinan karakter
        # end if
    # end for

    if Main.showSteps == True: # show steps #######################################################
        print("\nstep 2 - len(contours) = " + str(len(contours)))  # 2362 with MCLRNF1 image
        print("step 2 - intCountOfPossibleChars = " + str(intCountOfPossibleChars))  # 131 with MCLRNF1 image
        cv2.imshow("2a", imgContours)
    # end if # show steps #########################################################################

    return listOfPossibleChars
# end function


###################################################################################################
def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()           # ini akan menjadi nilai pengembalian

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # urutkan karakter dari kiri ke kanan berdasarkan posisi x

            # hitung titik pusat plat
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

            # hitung lebar dan tinggi plat
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    # berakhir untuk

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

            # hitung sudut koreksi daerah plat
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

            # titik pusat wilayah pelat paket, lebar dan tinggi, dan sudut koreksi ke dalam variabel anggota pelat yang diputar
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

            # langkah terakhir adalah melakukan rotasi yang sebenarnya

            # dapatkan matriks rotasi untuk sudut koreksi yang dihitung
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape      # membongkar lebar dan tinggi gambar asli

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))       # putar seluruh gambar

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possiblePlate.imgPlate = imgCropped         # salin gambar plat yang dipangkas ke dalam variabel anggota yang berlaku dari plat yang mungkin

    return possiblePlate
# fungsi akhir












