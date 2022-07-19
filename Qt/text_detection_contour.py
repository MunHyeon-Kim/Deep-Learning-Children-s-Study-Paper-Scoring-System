import cv2
import numpy as np
import os

def contour (large):
    rgb = large
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 9))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours, hierarchy, _ = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(bw.shape, dtype=np.uint8)
    return contours, mask, small, rgb

def detect(img_folder_dir, result_dir):
    files = os.listdir(img_folder_dir)
    i= -10
    detection_count = 0

    for file in files:
        img = cv2.imread(os.path.join(img_folder_dir, file))
        #cv2.imshow('aef',img)
        cv2.waitKey(0)
        contours, mask, small, rgb = contour(img)
        i += 10
        for idx in range(len(contours)):
            i += 1
            x, y, w, h = cv2.boundingRect(contours[idx])
            mask[y:y + h, x:x + w] = 0
            cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
            r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)
            if r > 0.0001 and w > 4 and h > 4:
                cv2.rectangle(rgb, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
            cropped_img = small[y: y + h, x: x + w]
            #cv2.imwrite("result{}.png".format(idx), cropped_img)
            cv2.imwrite(os.path.join(result_dir, "result_" + "result{}.png".format(i)), cropped_img)
            detection_count += 1

            #cv2.imshow('rects', rgb)
            #cv2.waitKey()



    # show image with contours rect
    cv2.imshow('rects', rgb)
    cv2.waitKey()
    return detection_count

def detect2(img_dir, result_dir):
    """이미지 하나 detection"""
    result_abs_dirs = []

    img = cv2.imread(img_dir)
    img_file_name = os.path.split(img_dir)[-1]
    img_file_name_WO_ext, _ = os.path.splitext(img_file_name)
    #cv2.imshow('aef',img)
    cv2.waitKey(0)
    contours, mask, small, rgb = contour(img)
    i = 0
    for idx in range(len(contours)):
        i += 1
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y + h, x:x + w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)
        if r > 0.0001 and w > 4 and h > 4:
            cv2.rectangle(rgb, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
        cropped_img = small[y: y + h, x: x + w]
        #cv2.imwrite("result{}.png".format(idx), cropped_img)
        img_save_dir = os.path.join(result_dir, img_file_name_WO_ext + "_{}.png".format(i))
        cv2.imwrite(img_save_dir, cropped_img)
        result_abs_dirs.append(img_save_dir)

        #cv2.imshow('rects', rgb)
        #cv2.waitKey()



    # show image with contours rect
    # cv2.imshow('rects', rgb)
    # cv2.waitKey()
    return result_abs_dirs