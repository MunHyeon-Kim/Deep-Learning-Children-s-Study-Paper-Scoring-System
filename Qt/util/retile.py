import cv2
import numpy as np
import os


def contour(large):
    rgb = large
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 50))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(bw.shape, dtype=np.uint8)
    return contours, mask, small, rgb


files = os.listdir('train')

for file in files:
    img = cv2.imread(os.path.join("train", file))

    img = cv2.pyrUp(img)

    contours, mask, small, rgb = contour(img)

    src = img.copy()

    temp = img.copy()

    h, w, c = img.shape

    # white image
    for i in range(0, h):
        for j in range(0, w):
            temp[i, j] = 255
    temp2 = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    temp2 = cv2.resize(temp2, (h, w))

    #.imshow('original', rgb)
    #cv2.waitKey()

    nextX = 0

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y + h, x:x + w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)
        if r > 0.0001 and w > 4 and h > 4:
            cv2.rectangle(rgb, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
        cropped_img = small[y: y + h, x: x + w]

        # temp2[nextX:nextX+w,0:h] = rgb[x: x + w, y: y + h]
        nextX = x
        i = 0
        j = 0
        for i in range(x, x + w):
            for j in range(y, y + h):
                temp2[i - x + nextX, j - y] = small[j, i].copy()

        # cv2.imshow('temp', rgb)
        # cv2.waitKey()

        temp3 = temp2.copy()

        temp3 = cv2.pyrDown(temp3)

        temp3 = cv2.rotate(temp3, cv2.ROTATE_90_CLOCKWISE)
        temp3 = cv2.flip(temp3, 1)

        cv2.imwrite(os.path.join("custom_data/train", os.path.basename(file)), temp3)

