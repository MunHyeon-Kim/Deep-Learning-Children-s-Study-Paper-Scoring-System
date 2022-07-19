Qt 폴더
main - 실행시 GUI가 실행됩니다. 
GUI 사용은 GUI.mp4에 영상으로 있습니다. 
log_demo : custom_data(학습지데이터) test 파일 test결과입니다. 
text_detection_contour - 이미지에서 글씨를 추출하여 자르는 코드입니다. 
img_diff - 원본과 비교하여 글씨를 추출하는 코드입니다. 

Qt-util 
base_contour 원본과 정답이 입력된 파일을 받아 지정된 위치의 글씨만 추출하는 코드입니다. 
customdata - 학습지 이미지를 학습시키기 위한 전처리 코드입니다. 
retile - 이미지에서 글씨의 위치를 위로정렬 해주는 코드입니다. 
skeleton - 굵기를 1로 바꿔주는 코드입니다. 

Qt - recognition 
https://github.com/clovaai/deep-text-recognition-benchmark
코드를 수정하여 만들었습니다. 딥러닝 글씨인식 부분입니다. 
입출력이 원하는데로 나오게 수정하였고, 모델을 seNet으로 수정하였습니다. 
ai hub의 한글 데이터셋을 먼저 학습시킨 뒤 학습지 데이터로 fine tuning 하였습니다. 

