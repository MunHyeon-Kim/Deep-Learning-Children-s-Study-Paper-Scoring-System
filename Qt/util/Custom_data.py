import shutil

import os



with open('list.txt', 'r') as file:    # hello.txt 파일을 읽기 모드(r)로 열기
    lines = file.readlines()

path = "C:/Users/huring/Desktop/custon_data/data/"
dest = "custom_data/images/"
images = os.listdir(path)

lines2 = []

for line in lines:
    lines2.append(line.replace("\n",""))

print(images)

print(lines2)

for image in images:
    if image in lines2:
        shutil.copyfile(path+image, dest+image)
        print(image)
