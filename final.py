import cv2
import numpy as np
import tensorflow as tf
from keras.api.models import load_model
import HandTranckingModule as htm
import os

# Charger le modèle de reconnaissance de caractères
model = load_model('best_model_best4.keras')

# Dictionnaire de correspondance des prédictions
letters = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l',
           12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w',
           23: 'x', 24: 'y', 25: 'z', 26: ''}
folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    if image is not None:
        overlayList.append(image)


    header = overlayList[0]
    drawColor=(87, 113, 255)

# Paramètres pour la capture vidéo
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Détecteur de main
detector = htm.handDetector(detectionCon=0.85)

# Variables pour le dessin
brushThickness = 25
eraserThickness = 100

xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
points = []
prediction = 26  # Initial prediction

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Détecter les mains
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()

        # Mode de sélection (deux doigts levés)
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 125:
                if 250 < x1 < 450:
                    drawColor = (87, 113, 255)
                elif 550 < x1 < 750:
                    drawColor = (0, 255, 0)
                elif 800 < x1 < 950:
                    drawColor = (255, 0, 0)
                elif 1050 < x1 < 1200:
                    drawColor = (0, 0, 0)

        # Mode de dessin (index levé)
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            points.append((x1, y1))
            xp, yp = x1, y1

    else:
        if len(points) != 0:
            blackboard_gray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            blur = cv2.medianBlur(blackboard_gray, 15)
            blur = cv2.GaussianBlur(blur, (5, 5), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            blackboard_cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(blackboard_cnts) >= 1:
                cnt = sorted(blackboard_cnts, key=cv2.contourArea, reverse=True)[0]
                if cv2.contourArea(cnt) > 1000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    alphabet = blackboard_gray[y - 10:y + h + 10, x - 10:x + w + 10]
                    try:
                        img_resized = cv2.resize(alphabet, (28, 28))
                    except cv2.error as e:
                        points = []
                        imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                        continue
                    img_resized = np.array(img_resized)
                    img_resized = img_resized.astype('float32') / 255
                    prediction = model.predict(img_resized.reshape(1, 28, 28, 1))[0]
                    prediction = np.argmax(prediction)
                    points = []
                    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    img[0:125, 0:1280] = header
    # Affichage des dessins et de l'image
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Afficher la prédiction sur l'image
    cv2.putText(img, "Prediction: " + str(letters[prediction]), (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                2)

    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
