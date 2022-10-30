import math

import cv2
import mediapipe as mp
import time
import pycaw
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)
    lis = []
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for index, lm in enumerate(hand.landmark):
                h ,w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if index == 4:
                    x1, y1 = cx, cy
                    cv2.circle(img, (x1, y1), 15, (0, 0, 0), cv2.FILLED)
                if index == 8:
                    x2, y2 = cx, cy
                    cv2.circle(img, (x2, y2), 15, (0, 0, 0), cv2.FILLED)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cxx, cyy = (x1+x2)//2 , (y1+y2)//2
                    cv2.circle(img, (cxx, cyy), 15, (0, 0, 0), cv2.FILLED)
                    length = math.hypot(x2-x1, y2-y1)

                    vol = np.interp(length, [30,200], [minVol,maxVol])
                    volume.SetMasterVolumeLevel(vol, None)
                    if length < 40:
                        cv2.circle(img, (cxx, cyy), 15, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)

            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    if cv2.getWindowProperty("Image", 0) == -1:
        cv2.destroyAllWindows()
        break