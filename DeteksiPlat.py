import cv2
import numpy as np
import os
import xlsxwriter
import Main
import time

time.sleep(1)

def saveDatabase():
    print(Main.platDetected)
    wb = xlsxwriter.Workbook("Plat Terdeteksi.xlsx")
    ws = wb.add_worksheet()
    row = 0
    col = 0
    for item in Main.platDetected:

        ws.write(row, col, item)
        row += 1
    wb.close()
    os.system("Plat Terdeteksi.xlsx")


testVideo = False


if testVideo == False:
    gambar = cv2.imread("12.jpg")
    Main.main(gambar)
    saveDatabase()
elif testVideo == True:
    cam = cv2.VideoCapture("video90.mp4")
    currentframe = 0
    while True:
        currentframe += 1
        ret,frame = cam.read()
        if ret:
            imgResize = cv2.resize(frame, None, fx = 2, fy = 2, interpolation = cv2.INTER_LINEAR)
            cv2.imshow("plat", imgResize)
            time.sleep(0.05)
            if (currentframe % 30) == 0:
                Main.main(imgResize)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                saveDatabase()
                break
        else:
            saveDatabase()
            break
    cam.release()
    cv2.destroyAllWindows()


