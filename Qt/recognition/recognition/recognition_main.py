# -*- coding: utf-8 -*-
# import text_detection_cv as td
# from text_recogition_naverAi import demo as text_recogition_naverAi
import os
import re
import cv2
import string
import argparse
import torch.backends.cudnn as cudnn
import torch.utils.data
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

# import demo2

from text_recogition_naverAi.recognition import demo2

mode = 0
base_img_dir = f'base_img/'
ans_img_dir = f'answer_image/'
ans_txt_dir = f'answer_txt'


# 글자 추출
# 이미지가 들어오면 ( 폴더는 for문 돌리면되고 )
# 본인이랑 이름이 같은 베이스 이미지 폴더를 돌려서 찾아서 넣고
# 본인 이름이랑 같은 폴더에 저장.

# 입력 이미지경로, idx번째 이미지 -> 출력 배경 지운 이미지
def diff(img, idx):

    dd = os.listdir(img)

    name = os.path.basename(dd[idx])
    temp = name
    name = re.sub(' [(].*[)]', '', name)

    base_img_list = os.listdir(base_img_dir)
    for base in base_img_list:
        if base == name:
            img2 = cv2.imread(img+dd[idx])
            base = cv2.imread(base_img_dir+base)
            diffed = td.img_diff(img2, base)
            return diffed, dd[idx]


# 이미지 커팅
# detection 돌려서 범위 대로 커팅하는데
# 정답 이미지를 불러와야대고.

# 입력 : 이미지 -> 출력 커팅이미지(리스트), 원본파일이름
def cutting(img , name):
    name = os.path.basename(name)
    name = re.sub(' [(].*[)]', '', name)

    ans_img_list = os.listdir(ans_img_dir)

    for base in ans_img_list:
        if base == name:
            base = cv2.imread(ans_img_dir + base)
            cutted, contours = td.detection(base, img)
            return cutted, contours


# 이미지 인식


def run():
    # if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--image_folder', default="debug_result",
                            help='path to image_folder which contains text images')
        parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
        parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
        parser.add_argument('--saved_model', default='text_recogition_naverAi/text_recogition_naverAi/pretrained_models/best_accuracy.pth',
                            help="path to saved_model to evaluation")
        """ Data processing """
        parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
        parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
        parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
        parser.add_argument('--rgb', action='store_true', help='use rgb input')
        parser.add_argument('--character', type=str,
                            default='0123456789abcdefghijklmnopqrstuvwxyz가각간갇갈감갑값갓강갖같갚갛개객걀걔거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀귓규균귤그극근글긁금급긋긍기긴길김깅깊까깍깎깐깔깜깝깡깥깨꺼꺾껌껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꾼꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냇냉냐냥너넉넌널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐댓더덕던덜덟덤덥덧덩덮데델도독돈돌돕돗동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿링마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몬몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭘뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벨벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브븐블비빌빔빗빚빛빠빡빨빵빼뺏뺨뻐뻔뻗뼈뼉뽑뿌뿐쁘쁨사삭산살삶삼삿상새색샌생샤서석섞선설섬섭섯성세섹센셈셋셔션소속손솔솜솟송솥쇄쇠쇼수숙순숟술숨숫숭숲쉬쉰쉽슈스슨슬슴습슷승시식신싣실싫심십싯싱싶싸싹싼쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓴쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액앨야약얀얄얇양얕얗얘어억언얹얻얼엄업없엇엉엊엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷옹와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡잣장잦재쟁쟤저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쩔쩜쪽쫓쭈쭉찌찍찢차착찬찮찰참찻창찾채책챔챙처척천철첩첫청체쳐초촉촌촛총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칫칭카칸칼캄캐캠커컨컬컴컵컷케켓켜코콘콜콤콩쾌쿄쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택터턱턴털텅테텍텔템토톤톨톱통퇴투툴툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔팝패팩팬퍼퍽페펜펴편펼평폐포폭폰표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홈홉홍화확환활황회획횟횡효후훈훌훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘?!.,()',
                            help='character label')
        parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
        parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
        """ Model Architecture """
        parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
        parser.add_argument('--FeatureExtraction', type=str, default='ResNet',
                            help='FeatureExtraction stage. VGG|RCNN|ResNet')
        parser.add_argument('--SequenceModeling', type=str, default='BiLSTM',
                            help='SequenceModeling stage. None|BiLSTM')
        parser.add_argument('--Prediction', type=str, default='CTC', help='Prediction stage. CTC|Attn')
        parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
        parser.add_argument('--input_channel', type=int, default=1,
                            help='the number of input channel of Feature extractor')
        parser.add_argument('--output_channel', type=int, default=512,
                            help='the number of output channel of Feature extractor')
        parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
        parser.add_argument('--label_test', default='ocr_data/gt_test.txt', help='path to labels.txt for test images')
        parser.add_argument('--log_filename', default='log.txt', help='log file name')

        opt = parser.parse_args()

        """ vocab / character number configuration """
        if opt.sensitive:
            opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        cudnn.benchmark = True
        cudnn.deterministic = True
        opt.num_gpu = torch.cuda.device_count()

        return demo2.demo(opt)
    except Exception as e:
        print(e)

# run()




#diffed, name  = diff('img/', 0)

#cut = cutting(diffed, name)
result = run()
if result != None:
    with open("recognition_result.txt", "w", encoding="UTF-8") as file:
        for value in result:
            file.write(value + "\n")