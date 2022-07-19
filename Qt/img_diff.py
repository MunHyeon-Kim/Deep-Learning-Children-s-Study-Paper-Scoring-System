import cv2
import numpy as np
import os

"""
base_img = cv2.imread("base_img.png")

files = os.listdir("img")

for file in files:
    img = cv2.imread(os.path.join("img", file))

    diff = cv2.absdiff(img, base_img)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    th = 1
    imask =  mask > th

    canvas = np.zeros_like(img, np.uint8) + 255
    canvas[imask] = img[imask]

    cv2.imwrite(os.path.join("absdiff_result", "result_" + os.path.basename(file)), canvas)
"""

def get_diff_img(img_dir, base_img_dir, save_folder):
    base_img = cv2.imread(base_img_dir)

    img = cv2.imread(img_dir)

    diff = cv2.absdiff(img, base_img)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    th = 1
    imask =  mask > th

    canvas = np.zeros_like(img, np.uint8) + 255
    canvas[imask] = img[imask]

    cv2.imwrite(os.path.join(save_folder, os.path.basename(img_dir)), canvas)
    print("diff_img_saved")
    return os.path.join(save_folder, os.path.basename(img_dir))