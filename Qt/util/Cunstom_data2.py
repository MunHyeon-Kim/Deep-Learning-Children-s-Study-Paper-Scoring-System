#-*- coding: utf-8 -*-
import shutil

import os
import random

path = 'custom_data/images/'
dest = 'custom_data/train/'
dest2 = 'custom_data/validation/'
dest3 = 'custom_data/test/'

images = os.listdir(path)

with open('list.txt', 'r') as file:
    lines = file.readlines()

with open('labeled.txt', 'r',encoding="UTF-8") as file:
    labels = file.readlines()

lines2 = []

for line in lines:
    lines2.append(line.replace("\n",""))

lines = lines2

print(len(lines))
print(len(labels))

c = list(zip(lines, labels))

random.shuffle(c)

lines, labels = zip(*c)

print(len(lines))

train_files = lines[:1600]
validation_files = lines[1600:1800]
test_files = lines[1800:2000]


for image in images:
    if image in train_files:
        shutil.copyfile(path+image, dest+image)
        print(image)

for image in images:
    if image in validation_files:
        shutil.copyfile(path+image, dest2+image)
        print(image)

for image in images:
    if image in test_files:
        shutil.copyfile(path+image, dest3+image)
        print(image)

train_files = labels[:1600]
validation_files = labels[1600:1800]
test_files = labels[1800:2000]

gt_train = []
for i in train_files:
    i = "train/" + i
    i.replace("png","png\t")
    gt_train.append(i)

gt_validation = []
for i in validation_files:
    i = "validation/" + i
    i.replace("png","png\t")
    gt_validation.append(i)

gt_test = []
for i in validation_files:
    i = "test/" + i
    i.replace("png","png\t")
    gt_test.append(i)


with open('custom_data/gt_train.txt', 'w', encoding='utf-8') as file:    # hello.txt 파일을 쓰기 모드(w)로 열기
    file.writelines(gt_train)

with open('custom_data/gt_validation.txt', 'w', encoding='utf-8') as file:    # hello.txt 파일을 쓰기 모드(w)로 열기
    file.writelines(gt_validation)

with open('custom_data/gt_test.txt', 'w', encoding='utf-8') as file:    # hello.txt 파일을 쓰기 모드(w)로 열기
    file.writelines(gt_test)