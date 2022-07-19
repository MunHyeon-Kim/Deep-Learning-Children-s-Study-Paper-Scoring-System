import cv2
import numpy as np
import os



files = os.listdir('custom_data/train')

for file in files:
    img = cv2.imread(os.path.join("custom_data/train", file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = 255 - img

    ret, img = cv2.threshold(img, 127, 255, 0)

    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    skel = 255 - skel
    cv2.imwrite(os.path.join("train_result2", os.path.basename(file)), skel)
