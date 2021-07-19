import cv2
import numpy as np
import time
import poseModule as pm

# cap = cv2.VideoCapture("video/kids.mp4")
cap = cv2.VideoCapture(0)

detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

lo = 30
hi = 165

while True:
    _, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    if len(lmList) != 0:
        # Right Arm
        angle = detector.findAngle(img, 12, 14, 16, True)

        # Left Arm
        # angle = detector.findAngle(img, 11, 13, 15, True)

        # Right Leg
        # angle = detector.findAngle(img, 24, 26, 28, True)

        # Left Leg
        # angle = detector.findAngle(img, 23, 25, 27, True)

        # Right Hip
        # angle = detector.findAngle(img, 11, 23, 25, True)

        # Left Hip
        # angle = detector.findAngle(img, 12, 24, 26, True)

        angle = 200 - angle
        per = np.interp(angle, (lo, hi), (0, 100))
        bar = np.interp(angle, (lo, hi), (650, 100))

        # Detect workout curls
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0
        # print(count)

        # Draw Bar
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 1)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, str(int(per))+'%', (1080, 75), cv2.FONT_HERSHEY_PLAIN, 3,
                    color, 2)

        # Draw Curl Count
        cv2.rectangle(img, (0, 600), (160, 720), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, str(int(count)), (4, 700), cv2.FONT_HERSHEY_PLAIN, 7,
                    (0, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 60), cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 0, 255), 2)

    cv2.imshow("AI Personal Trainer", img)
    cv2.waitKey(1)
