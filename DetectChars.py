# DetectChars.py
import os

import cv2
import numpy as np
import math
import random
import time

import Main
import Preprocess
import PossibleChar

#variabel tingkat modul ##########################################################################

kNearest = cv2.ml.KNearest_create()

        # konstanta untuk diperiksa jika memungkinkan karakter, ini hanya memeriksa satu kemungkinan karakter (tidak dibandingkan dengan karakter lain)
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

        # konstanta untuk membandingkan dua karakter
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

        # konstanta lainnya
MIN_NUMBER_OF_MATCHING_CHARS = 5

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100


folderName = "Proses/" + str(time.strftime("%H%M", time.localtime()))
###################################################################################################
def loadKNNDataAndTrainKNN():
    allContoursWithData = []                # nyatakan daftar kosong,
    validContoursWithData = []              # kami akan segera mengisinya

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # baca di klasifikasi pelatihan
    except:                                                                                 # jika file tidak bisa dibuka
        print("kesalahan, tidak dapat membuka klasifikasi.txt, keluar dari program\n")  # tampilkan pesan kesalahan
        os.system("berhenti sebentar")
        return False                                                                        # dan kembali salah
    # akhir coba

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # baca di gambar pelatihan
    except:                                                                                 # jika saya tidak bisa dibuka
        print("kesalahan, tidak dapat membuka flattened_images.txt, keluar dari program\n")  # tampilkan pesan kesalahan
        os.system("berhenti sebentar")
        return False                                                                        # dan kembali salah
    # akhir coba

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # membentuk kembali array numpy ke Id, perlu diteruskan ke panggilan ke kereta

    kNearest.setDefaultK(1)                                                             # atur default K ke I

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)           # melatih objek KNN

    return True                             # jika kita sampai di sini pelatihan berhasil jadi kembalilah benar
# end functionfungsi akhir

###################################################################################################
def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []
    if len(listOfPossiblePlates) == 0:          # jika daftar kemungkinan plat kosong
        return listOfPossiblePlates             # kembali
    # dan jika

            # pada titik ini kita dapat memastikan daftar plat yang mungkin memiliki setidaknya satu plat
    currentlist = 0
    for possiblePlate in listOfPossiblePlates:          # untuk setiap plat yang mungkin, ini adalah loop untuk besar yang mengambil sebagian besar fungsi
        currentlist += 1
        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocess.preprocess(possiblePlate.imgPlate)     # praproses untuk mendapatkan gambar skala abu-abu dan biner

        if Main.showSteps == True: # menunjukkan langkah-langkah ###################################################
            cv2.imshow("5a", possiblePlate.imgPlate)
            cv2.imshow("5b", possiblePlate.imgGrayscale)
            cv2.imshow("5c", possiblePlate.imgThresh)
        name5a = folderName + '/Negatif.jpg'
        cv2.imwrite( name5a, possiblePlate.imgPlate)
        name5b = folderName + '/Keabuan.jpg'
        cv2.imwrite( name5b, possiblePlate.imgGrayscale)
        name5c = folderName + '/Biner.jpg'
        cv2.imwrite( name5c, possiblePlate.imgThresh)
        # dan saya # menunjukkan langkah-langkah #####################################################################

                # tingkatkan ukuran gambar plat agar lebih mudah dilihat dan deteksi karakter
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)

                # biner batas lagi untuk menghilangkan area abu-abu
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if Main.showSteps == True: # menunjukkan langkah-langkah ###################################################
            cv2.imshow("5d", possiblePlate.imgThresh)
        # dan saya # menunjukkan langkah-langkah #####################################################################

                # temukan semua kemungkinan karakter di plat,
                # fungsi ini perta-tama menemukan semua kontur, lalu hanya menyatakan kontur yang dapat berupa karakter (belum di bandingkan dengan karakter lain)
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        #jika utama. menunjukkan langka-langkah == benar: # menunjukkan langkah-langkah ###################################################
        height, width, numChannels = possiblePlate.imgPlate.shape
        imgContours = np.zeros(((height*2), (width*2), 3), np.uint8)
        del contours[:]                                         # hapus data kontur

        for possibleChar in listOfPossibleCharsInPlate:
            contours.append(possibleChar.contour)
        # berakhir untuk

        cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)

        #cv2.imshow("6", kontur gambar)
        name6 = folderName + '/6' + str(currentlist) + '.jpg'
        cv2.imwrite( name6, imgContours)
        # dan saya # menunjukkan langkah-langkah #####################################################################

                # diberikan daftar semua karakter yang mungkin, temukan grup karakter yang cocok di dalam plat
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

        #if Main.showSteps == benar: # menunjukkan langkah-langkah ###################################################
        imgContours = np.zeros(((height*2), (width*2), 3), np.uint8)
        del contours[:]

        for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
            # end for
            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
        # end for
        #cv2.imshow("7", kontur gambar)
        name7 = folderName + '/Kontur.jpg'
        cv2.imwrite( name7, imgContours)
        # end if # menunjukkan langkah-langkah #####################################################################

        if (len(listOfListsOfMatchingCharsInPlate) == 0):			# jika tidak ada kelompok karakter yang cocok ditemukan di plat

            if Main.showSteps == True: # menunjukkan langkah-langkah ###############################################
                print("chars found in plate number " + str(
                    intPlateCounter) + " = (none), click on any image and press a key to continue . . .")
                intPlateCounter = intPlateCounter + 1
                cv2.destroyWindow("8")
                cv2.destroyWindow("9")
                cv2.destroyWindow("10")
                cv2.waitKey(0)
            # end if # menunjukkan langkah-langkah #################################################################

            possiblePlate.strChars = ""
            continue						# kembali ke atas untuk loop
        # end if

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):                              # dalam setiap daftar karakter yang cocok
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)        # urutkan karakter dari kiri dan kanan
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])              # dan hapus karakter yang tumpang tindi di dalam        # end for

        #if Main.showSteps == Benar: # menunjukkan langkah-langkah ###################################################
        imgContours = np.zeros(((height*2), (width*2), 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            del contours[:]

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
        # end for
        #cv2.imshow("8", kontur gambar)
        name8 = folderName + '/8.jpg'
        cv2.imwrite( name8, imgContours)
        # end if # menunjukkan langkah-langkah #####################################################################

                # dalam setiap plat yang mungkin, misalkan dafter karakter percocokan potensial terpanjang adalah daftar karakter yang sebenarnya
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

                # loop melalui semua vektor karakter yang cocok, dapatkan indeks yang memiliki karakter paling banyak
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
            # end if
        # berakhir untuk

                # misalkan daftar karakter yang cocok terpanjang di dalam plat adalah daftar karakter yang sebenarnya
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        #if Main.showSteps == Banar: # menunjukkan langkah-langkah ###################################################
        imgContours = np.zeros(((height*2), (width*2), 3), np.uint8)
        del contours[:]

        for matchingChar in longestListOfMatchingCharsInPlate:
            contours.append(matchingChar.contour)
        # end for

        cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)

        #cv2.imshow("9", imgContours)
        name9 = folderName + '/9.jpg'
        cv2.imwrite( name9, imgContours)
        # dan jika # menunjukkan langkah-langkah #####################################################################

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)

        if Main.showSteps == True: # menunjukkan langkah-langkah ###################################################
            print("chars found in plate number " + str(
                intPlateCounter) + " = " + possiblePlate.strChars + ", click on any image and press a key to continue . . .")
            intPlateCounter = intPlateCounter + 1
            #cv2.waitKey(0)
        # dan jika # menunjukkan langkah-langkah #####################################################################

    # ujung loop for besar yang mengambil sebagian besar fungsi

    if Main.showSteps == True:
        print("\nchar detection complete, click on any image and press a key to continue . . .\n")
        cv2.waitKey(0)
    # berakhir jika

    return listOfPossiblePlates
# fungsi akhir

###################################################################################################
def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []                        # ini akan menjadi nilai kembali
    contours = []
    imgThreshCopy = imgThresh.copy()

            # temukan semua kontur di plat
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:                        # untuk setiap kontur
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(possibleChar):              # jika kontur adalah karakter yang memungkinkan, perhatikan ini tidak dibandingkan dengan karakter lain (yet) . . .
            listOfPossibleChars.append(possibleChar)       # tambahkan ke daftar kemungkinan karakter
        # berakhir jika
    # end if

    return listOfPossibleChars
# fungsi akhir

###################################################################################################
def checkIfPossibleChar(possibleChar):
            # fungsi ini adalah pass pertama yang melakukan pemeriksaan kasar pada kontur untukmelihat apakha itu bisa menjadi karakter,
            # perhatikan bahwa kita tidak (yet) membandingkan karakter untuk mencari grup
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False
    # end if
# fungsi akhir

###################################################################################################
def findListOfListsOfMatchingChars(listOfPossibleChars):
            # dengan fungsi ini, kami memulai dengan semua karakter yang mungkindalam satu daftar besar
            # tujuan dari fungsi ini adalah untuk mengatur ulang satu daftar karakter menjadi daftar karakter yang cocok,
            # perhatikan bahwa karakter yang tidak ditemukan dalam grup kecocokan tidak perlu dipertimbangkan lebih lanjut
    listOfListsOfMatchingChars = []                  # ini akan menjadi nilai pengambilan

    for possibleChar in listOfPossibleChars:                        # untuk setiap karakter yang mungkan dalam satu daftar besar karakter
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)        # temukan semua karakter dalam daftar besar yang cocok dengan karakter saat ini

        listOfMatchingChars.append(possibleChar)                # juka tambahkan karakter saat ini ke daftarkemungkinan karakter yang cocok saat ini

        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     # jika kemungkinan daftar karakter yang cocok saat ini tidak cukup panjang untuk membentuk plat yang mungkin
            continue                            # lompat kembali ke atas for loop dan coba lagi dengan karakter berikutnya, perhatikan bahwa itu tidak perlu
                                                # untuk menyimpan daftar dengan cara apa pun karena tidak memiliki cukup karakter untuk menjadi plat yang memungkinkan
        # end if

                                                # jika kita sampai di sini, daftar saat ini lulus tes sebagai "grup" atau "cluster" dari karakter yang cocok
        listOfListsOfMatchingChars.append(listOfMatchingChars)      # jadi tambahkan ke daftar daftar karakter yang cocok

        listOfPossibleCharsWithCurrentMatchesRemoved = []

                                                # hapus daftar karakter yang cocok saat ini dari daftar besar sehingga kami tidak menggunakan karakter yang sama dua kali,
                                                # pastikan untuk membuat daftar besar baru untuk ini karena kami tidak ingin mengubah daftar besar asli
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      # panggilan rekursif

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        # untuk setiap daftar karakter yang cocok ditemukan oleh panggilan rekursif
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             # tambahkan ke daftar asli kami dari daftar karakter yang cocok
        # end for

        break       # exit for

    # end for

    return listOfListsOfMatchingChars
# fungsi akhir

###################################################################################################
def findListOfMatchingChars(possibleChar, listOfChars):
            # tujuan dari fungsi ini adalah, mengingat kemungkinan karakter dan daftar besar kemungkinan karakter,
            # temukan semua karakter dalam daftar besar yang cocok untuk satu kemungkinan karakter, dan kembalikan karakter yang cocok itu sebagai daftar
    listOfMatchingChars = []                # ini akan menjadi nilai pengembalian


    for possibleMatchingChar in listOfChars:                # untuk setiap karakter dalam daftar besar
        if possibleMatchingChar == possibleChar:    # jika karakter yang kami coba temukan kecocokan adalah karakter yang sama persis dengan karakter dalam daftar besar yang sedang kami periksa
                                                    # maka kita tidak boleh memasukkannya ke dalam daftar kecocokan b/c yang akan berakhir ganda termasuk karakter saat ini
            continue                                # jadi jangan tambahkan ke daftar kecocokan dan lompat kembali ke atas for loop
        # end if
                    # menghitung barang untuk melihat apakah karakter cocok
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

                # periksa apakah karakter cocok


        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar)        # jika karakternya cocok, tambahkan karakter saat ini ke daftar karakter yang cocok
        # end if
    # end for

    return listOfMatchingChars                  # hasil kembali
# end function

###################################################################################################
# gunakan teorema Pythagoras untuk menghitung jarak antara dua karakter
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))
# end function

###################################################################################################
# gunakan trigonometri dasar (SOH CAH TOA) untuk menghitung sudut antar karakter
def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:                           # periksa untuk memastikan kita tidak membagi dengan nol jika posisi X tengah sama, pembagian float dengan nol akan menyebabkan crash di Python
        fltAngleInRad = math.atan(fltOpp / fltAdj)      # jika berdekatan bukan nol, hitung sudut
    else:
        fltAngleInRad = 1.5708                          # jika berdekatan adalah nol, gunakan ini sebagai sudut, ini agar konsisten dengan versi C++ dari program ini
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       # hitung sudut dalam derajat

    return fltAngleInDeg
# fungsi akhir

###################################################################################################
# jika kita memiliki dua karakter yang tumpang tindih atau saling berdekatan agar mungkin menjadi karakter yang terpisah, hapus karakter bagian dalam (lebih kecil),
# ini untuk mencegah memasukkan karakter yang sama dua kali jika dua kontur ditemukan untuk karakter yang sama,
# misalnya untuk huruf 'O' baik cincin bagian dalam dan cincin luar dapat ditemukan sebagai kontur, tetapi kami hanya memasukkan karakter sekali
def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)                # ini akan menjadi nilai pengembalian

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:        # jika karakter saat ini dan karakter lainnya tidak sama karakter . . .
                                                                            # jika karakter saat ini dan karakter lainnya memiliki titik pusat di lokasi yang hampir sama . . .
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                                # jika kita masuk ke sini, kita telah menemukan karakter yang tumpang tindih
                                # selanjutnya kita mengidentifikasi char mana yang lebih kecil, kemudian jika karakter itu belum dihapus pada pass sebelumnya, hapus
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:         # jika karakter saat ini lebih kecil dari karakter lain
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:              # jika karakter saat ini belum dihapus pada pass sebelumnya . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)         # lalu hapus karakter saat ini
                        # berakhir jika
                    else:                                                                       # lain jika karakter lain lebih kecil dari karakter saat ini
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:                # jika karakter lain belum dihapus pada pass sebelumnya . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)           # lalu hapus karakter lainnya
                        # berakhir jika
                    # berakhir jika
                # berakhir jika
            # berakhir jika
        # berakhir jika
    # berakhir jika

    return listOfMatchingCharsWithInnerCharRemoved
# fungsi akhir

###################################################################################################
# di sinilah kami menerapkan pengenalan karakter yang sebenarnya
def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""               # ini akan menjadi nilai balik, karakter di plat lic

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # urutkan karakter dari kiri ke kanan

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)                     # buat versi warna dari gambar ambang sehingga kita bisa menggambar kontur berwarna di atasnya

    for currentChar in listOfMatchingChars:                                         # untuk setiap karakter di plat
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2, Main.SCALAR_GREEN, 2)           # menggambar kotak hijau di sekitar karakter

                # pangkas karakter keluar dari gambar biner
        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))           # mengubah ukuran gambar, ini diperlukan untuk pengenalan karakter

        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))        # ratakan gambar menjadi array numpy 1d

        npaROIResized = np.float32(npaROIResized)               # konversi dari array int numpy 1d ke array float numpy 1d

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)              # akhirnya kita bisa memanggil findNearest !!!

        strCurrentChar = str(chr(int(npaResults[0][0])))            # dapatkan karakter dari hasil

        strChars = strChars + strCurrentChar                        # tambahkan karakter saat ini ke string penuh

    # berakhir untuk

    if Main.showSteps == True: # menunjukkan langkah-langkah #######################################################
        cv2.imshow("10", imgThreshColor)
    name10 = folderName + '/vektor.jpg'
    cv2.imwrite( name10, imgThreshColor)
    # berakhir jika # menunjukkan langkah-langkah #########################################################################

    return strChars
# fungsi akhir








