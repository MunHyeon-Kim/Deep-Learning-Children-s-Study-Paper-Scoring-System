import cv2
import numpy as np
import os


def img_diff(img, base_img):
    diff = cv2.absdiff(img, base_img)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    th = 1
    imask = mask > th

    canvas = np.zeros_like(img, np.uint8) + 255
    canvas[imask] = img[imask]

    return canvas


def contour(large):
    rgb = large
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(bw.shape, dtype=np.uint8)

    return contours, mask, small, rgb


def base_image2(large):
    rgb = large
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 15))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours


def detection(base_image, img):
    crapped = []
    idx = 0

    contour_main = base_image2(base_image)
    contours, mask, small, rgb = contour(img)
    crapped.append(idx)
    idx += 1

    for i in range(len(contour_main)):
        x, y, w, h = cv2.boundingRect(contour_main[i])
        for j in range(len(contours)):
            x2, y2, w2, h2 = cv2.boundingRect(contours[j])
            if range(x - int(w / 2), x + int(w / 2)).count(x2) == 0:
                continue
            elif range(y - int(h / 2), y + int(h / 2)).count(y2) == 0:
                continue
            else:
                mask[y:y + h, x:x + w] = 0
                cv2.drawContours(mask, contours, j, (255, 255, 255), -1)
                r = float(cv2.countNonZero(mask[y2:y2 + h2, x2:x2 + w])) / (w2 * h2)
                if r > 0.0001 and w2 > 4 and h2 > 4:
                    cv2.rectangle(rgb, (x2, y2), (x2 + w2 - 1, y2 + h2 - 1), (0, 255, 0), 2)

                cropped_img = small[y2: y2 + h2, x2: x2 + w2]
                crapped.append(cropped_img)
                cv2.imwrite("test{}.png".format(j + idx), cropped_img)
                break

        # show image with contours rect
        # cv2.imshow('rects', rgb)
        # cv2.waitKey()

    return crapped, contours



