# This Python file uses the following encoding: utf-8

import enum
import os
import shutil
import sys
from PySide2.QtCore import QFile, QObject, QThread, Signal
from PySide2.QtGui import QColor, QFont, QIcon, QPixmap
from PySide2.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QProgressDialog, QTableWidgetItem, QLabel, QPushButton
from resource.ui_main import Ui_MainWindow

import text_detection_contour
import img_diff

import time

import re
import cv2
import string
import argparse
import torch.backends.cudnn as cudnn
import torch.utils.data

from text_recogition_naverAi.recognition.utils import CTCLabelConverter, AttnLabelConverter
from text_recogition_naverAi.recognition.dataset import RawDataset, AlignCollate
from text_recogition_naverAi.recognition.model import Model

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

# pyside2-uic resource/ui_main.ui -o resource/ui_main.py     --> ui파일을 py로 변환

true_color = QColor(111, 237, 97)
false_color = QColor(237, 97, 97)


class MessageBoxType(enum.Enum):
    ABOUTQT = 0
    ABOUT = 1
    INFORMATION = 2
    QUESTION = 3
    WARNING = 4
    CRITICAL = 5


def run_recognition(img_folder):
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--image_folder', default=img_folder,
                            help='path to image_folder which contains text images')
        parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
        parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
        parser.add_argument('--saved_model', default='text_recogition_naverAi/best_accuracy2.pth',
                            help="path to saved_model to evaluation")
        """ Data processing """
        parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
        parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
        parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
        parser.add_argument('--rgb', action='store_true', help='use rgb input')
        parser.add_argument('--character', type=str,
                            default='0123456789abcdefghijklmnopqrstuvwxyz가각간갇갈감갑값갓강갖같갚갛개객걀걔거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀귓규균귤그극근글긁금급긋긍기긴길김깅깊까깍깎깐깔깜깝깡깥깨꺼꺾껌껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꾼꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냇냉냐냥너넉넌널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐댓더덕던덜덟덤덥덧덩덮데델도독돈돌돕돗동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿링마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몬몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭘뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벨벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브븐블비빌빔빗빚빛빠빡빨빵빼뺏뺨뻐뻔뻗뼈뼉뽑뿌뿐쁘쁨사삭산살삶삼삿상새색샌생샤서석섞선설섬섭섯성세섹센셈셋셔션소속손솔솜솟송솥쇄쇠쇼수숙순숟술숨숫숭숲쉬쉰쉽슈스슨슬슴습슷승시식신싣실싫심십싯싱싶싸싹싼쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓴쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액앨야약얀얄얇양얕얗얘어억언얹얻얼엄업없엇엉엊엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷옹와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡잣장잦재쟁쟤저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쩔쩜쪽쫓쭈쭉찌찍찢차착찬찮찰참찻창찾채책챔챙처척천철첩첫청체쳐초촉촌촛총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칫칭카칸칼캄캐캠커컨컬컴컵컷케켓켜코콘콜콤콩쾌쿄쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택터턱턴털텅테텍텔템토톤톨톱통퇴투툴툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔팝패팩팬퍼퍽페펜펴편펼평폐포폭폰표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홈홉홍화확환활황회획횟횡효후훈훌훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘?!.,()', help='character label')
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

        return demo(opt)

def demo(opt):
    """ model configuration """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    predlist = []
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            log = open(opt.log_filename, 'a')
            dashed_line = '-' * 120
            head = f'{"image_path":25s}\t{"predicted_labels":15s}\tconfidence score\tcharacter error rate'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            total_err = 0
            total_len = 0
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
    
                predlist.append(pred)
                print(f'{img_name:25s}\t{pred:15s}\t{confidence_score:0.4f}')
                log.write(f'{img_name:25s}\t{pred:15s}\t{confidence_score:0.4f}')

            log.close()

    return predlist


# https://stackoverflow.com/a/51061279
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # self.setWindowIcon(ICON_PATH)
        self.setWindowTitle("채점 시스템")

        self.init_variables()
        self.init_widgets()


    def init_variables(self):
        """클래스 변수 초기화"""
        self.cwd = os.getcwd()
        # DEBUG
        print(self.cwd)


    def init_widgets(self):
        """위젯 시그널 추가"""
        # Correct Answer Uploading Tab
        self.ui.CAU_file_select_pushButton.clicked.connect(
            lambda: self.open_file_chooser(
                self.ui.CAU_file_select_lineEdit,
                "텍스트 또는 이미지 파일 (*.txt *.png)"
            )
        )
        self.ui.CAU_uploading_pushButton.clicked.connect(
            lambda: self.upload_correct_answer(
                self.ui.CAU_file_select_lineEdit
            )
        )
        self.ui.CAU_file_select_pushButton_2.clicked.connect(
            lambda: self.open_file_chooser(
                self.ui.CAU_file_select_lineEdit_2,
                "이미지 파일 (*.png)"
            )
        )
        self.ui.CAU_uploading_pushButton_2.clicked.connect(
            lambda: self.upload_original_image(
                self.ui.CAU_file_select_lineEdit_2
            )
        )

        # Grading Tab
        self.ui.ASG_file_select_pushButton.clicked.connect(
            lambda: self.open_file_chooser(
                self.ui.ASG_file_select_lineEdit,
                "이미지 파일 (*.png)"
            )
        )
        self.ui.ASG_grading_pushButton.clicked.connect(
            lambda: self.grade_answer_sheet(
                self.ui.ASG_file_select_lineEdit,
                self.ui.ASG_grading_status_tableWidget
            )
        )

        # Check Wrong Answer Tab
        self.ui.CGR_pushButton.clicked.connect(
            lambda: self.get_grade_results(
                self.ui.CGR_tableWidget
            )
        )


    def open_file_chooser(self, lineEdit, search_filter="*", linked_lineEdit=None):
        """파일 탐색기 다이얼로그 실행"""
        # Dialog에 처음으로 뜰 폴더 지정(기본값)
        open_dir = self.cwd
        # lineEdit에 이미 입력되어있는 문자 가져옴
        lineEdit_text = str(lineEdit.text()).strip()
        # 입력되어있는 문자가 파일 경로인 경우 그 경로로 지정
        if len(lineEdit_text) > 0 and os.path.isfile(lineEdit_text):
            open_dir = os.path.split(lineEdit_text)[0]
        # 입력되어있는 문자가 파일 여러개인 경우 하나만 보고 그 경로로 지정
        else:
            first_text = lineEdit_text.split(",")[0]
            first_text = first_text.strip()
            if len(first_text) > 0 and os.path.isfile(first_text):
                open_dir = os.path.split(first_text)[0]
        # 연괸되어있는 lineEdit이 있으면 거기에서 경로 가져와서 지정함
        if linked_lineEdit != None:
            linked_lineEdit_text = str(linked_lineEdit.text()).strip()
            if len(linked_lineEdit_text) > 0 and os.path.isfile(linked_lineEdit_text):
                open_dir = os.path.split(linked_lineEdit_text)[0]

        # Dialog 실행
        selected_files, _ = QFileDialog.getOpenFileNames(self, filter=search_filter, dir=open_dir)

        # 취소 누른 경우
        if len(selected_files) < 1:
            return
        # 파일 선택 완료한 경우
        if lineEdit != None:
            if len(selected_files) == 1:
                lineEdit.setText(selected_files[0])
            else:
                # lineEdit.setText("파일 " + str(len(selected_files)) + "개")
                line_edit_str = ""
                for i in range(len(selected_files) - 1):
                    line_edit_str += '"' + (str(selected_files[i]) + '", ')
                line_edit_str += '"' + str(selected_files[-1]) + '"'
                lineEdit.setText(line_edit_str)
        

    def show_messageBox(self, messageBox_type=MessageBoxType.ABOUT, text="", title=""):
        """Message Box 실행"""
        if messageBox_type == MessageBoxType.ABOUTQT:
            QMessageBox.aboutQt(self)
        if messageBox_type == MessageBoxType.ABOUT:
            QMessageBox.about(self, title, text)
        if messageBox_type == MessageBoxType.INFORMATION:
            QMessageBox.information(self, title, text, QMessageBox.Ok)
        if messageBox_type == MessageBoxType.QUESTION:
            QMessageBox.question(self, title, text, QMessageBox.Ok, QMessageBox.Cancel)
        if messageBox_type == MessageBoxType.WARNING:
            QMessageBox.warning(self, title, text, QMessageBox.Ok)
        if messageBox_type == MessageBoxType.CRITICAL:
            QMessageBox.critical(self, title, text, QMessageBox.Ok)


    def lineEdit_text_to_file_name(self, text):
        """lineEdit의 텍스트를 파일 이름 리스트로 반환"""
        if len(text) < 1:
            # lineEdit이 비어있는 경우 종료
            return
        # 문자열에 존재하는 따옴표(") 제거
        text = text.replace('"', '')
        # ","를 기준으로 나눔
        text = text.split(",")
        return [s.strip() for s in text]


    def upload_correct_answer(self, lineEdit):
        """선택한 정답 업로드"""
        selected_files = self.lineEdit_text_to_file_name(lineEdit.text())
        
        # 정답 저장할 경로 없는 경우 생성
        data_abs_path = os.path.join(self.cwd, "data")
        if not os.path.exists(data_abs_path):
            os.mkdir(data_abs_path)
        dst_abs_path = os.path.join(data_abs_path,"correct_answers")
        if not os.path.exists(dst_abs_path):
            os.mkdir(dst_abs_path)
        # 파일 복사
        successful = True
        error_message = "정답 업로드 실패"
        for file in selected_files:
            # 현재 요소가 file이 아닌 경우
            if not os.path.isfile(file):
                successful = False
                # 에러 메시지 작성
                error_message += ("\n" + file)
                continue
            
            # 파일이 맞는 경우
            _, ext = os.path.splitext(file)
            # 파일이 텍스트인 경우
            if ext == ".txt":
                shutil.copy(file, dst_abs_path)
            # 파일이 이미지인 경우
            elif ext == ".png":
                original_image_abs_path = os.path.join(data_abs_path,"original_images")
                original_img_list = os.listdir(original_image_abs_path)
                sheet_name = os.path.splitext(os.path.basename(file))[0].strip()
                base_name = ""
                for s in sheet_name.split("_")[:-1]:
                    base_name += (s + "_")
                base_name = base_name[:-1]
                print(base_name)
                if not os.path.exists(original_image_abs_path) or base_name + ".png" not in original_img_list:
                    successful = False
                    # 에러 메시지 작성
                    error_message += ("\n" + file + " (원본 이미지 없음)")
                    continue
                # TODO: 필기체 인식 후 결과 리스트로 받아서 저장
                # img diff
                temp_diff_img_path = os.path.join(data_abs_path, "temp_diff_img")
                if not os.path.exists(temp_diff_img_path):
                    os.mkdir(temp_diff_img_path)
                save_diff_dir = img_diff.get_diff_img(file, os.path.join(original_image_abs_path, base_name + ".png"), temp_diff_img_path)

                # 필기체 detection
                temp_detection_abs_path = os.path.join(data_abs_path, "temp_detected_img")
                if not os.path.exists(temp_detection_abs_path):
                    os.mkdir(temp_detection_abs_path)
                detected_img_dirs = text_detection_contour.detect2(save_diff_dir, temp_detection_abs_path)

                # text_recogition_naverAi
                temp_recognition_result = run_recognition(temp_detection_abs_path)
                print(temp_recognition_result)
                
                # clean temp folders
                shutil.rmtree(temp_diff_img_path)
                shutil.rmtree(temp_detection_abs_path)
                with open(os.path.join(dst_abs_path, os.path.splitext(os.path.split(file)[1])[0]) + ".txt", "w", encoding="UTF-8") as new_file:
                    for line in temp_recognition_result:
                        new_file.write(line + "\n")
            # 지원하지 않는 파일 형식인 경우
            else:
                successful = False
                # 에러 메시지 작성
                error_message += ("\n" + file + " (지원하지 않는 파일 형식)")
            
        # 결과 알림
        if not successful:
            self.show_messageBox(messageBox_type=MessageBoxType.CRITICAL, text=error_message, title="Info")
        self.show_messageBox(messageBox_type=MessageBoxType.INFORMATION, text="업로드 완료", title="Info")

    def upload_original_image(self, lineEdit):
        """원본 이미지 업로드"""
        selected_files = self.lineEdit_text_to_file_name(lineEdit.text())
        
        # 정답 저장할 경로 없는 경우 생성
        data_abs_path = os.path.join(self.cwd, "data")
        if not os.path.exists(data_abs_path):
            os.mkdir(data_abs_path)
        dst_abs_path = os.path.join(data_abs_path,"original_images")
        if not os.path.exists(dst_abs_path):
            os.mkdir(dst_abs_path)
        # 파일 복사
        successful = True
        error_message = "정답 업로드 실패"
        for file in selected_files:
            # 현재 요소가 file이 아닌 경우
            if not os.path.isfile(file):
                successful = False
                # 에러 메시지 작성
                error_message += ("\n" + file)
                continue
            
            # 파일이 맞는 경우
            _, ext = os.path.splitext(file)
            # 파일이 이미지인 경우
            if ext == ".png":
                shutil.copy(file, dst_abs_path)
            # 지원하지 않는 파일 형식인 경우
            else:
                successful = False
                # 에러 메시지 작성
                error_message += ("\n" + file)
            
        # 결과 알림
        if not successful:
            self.show_messageBox(messageBox_type=MessageBoxType.CRITICAL, text=error_message, title="Info")
        self.show_messageBox(messageBox_type=MessageBoxType.INFORMATION, text="업로드 완료", title="Info")

    def grade_answer_sheet(self, lineEdit, tableWidget):
        """선택한 답안 업로드"""
        # lineEdit의 텍스트 가져옴
        selected_files = self.lineEdit_text_to_file_name(lineEdit.text())

        # 답안 저장할 경로 없는 경우 생성
        data_abs_path = os.path.join(self.cwd, "data")
        if not os.path.exists(data_abs_path):
            os.mkdir(data_abs_path)
        diff_image_abs_path = os.path.join(data_abs_path, "diff_images")
        if not os.path.exists(diff_image_abs_path):
            os.mkdir(diff_image_abs_path)
        original_image_abs_path = os.path.join(data_abs_path,"original_images")
        if not os.path.exists(original_image_abs_path):
            os.mkdir(original_image_abs_path)
        correct_answer_abs_path = os.path.join(data_abs_path, "correct_answers")
        if not os.path.exists(correct_answer_abs_path):
            os.mkdir(correct_answer_abs_path)
        grade_result_abs_path = os.path.join(data_abs_path, "grade_results")
        if not os.path.exists(grade_result_abs_path):
            os.mkdir(grade_result_abs_path)
        dst_abs_path = os.path.join(data_abs_path, "answer_sheets")
        if not os.path.exists(dst_abs_path):
            os.mkdir(dst_abs_path)
        detection_abs_path = os.path.join(dst_abs_path, "detection_result")
        if not os.path.exists(detection_abs_path):
            os.mkdir(detection_abs_path)
        recognition_abs_path = os.path.join(dst_abs_path, "text_recogition_naverAi")
        if not os.path.exists(recognition_abs_path):
            os.mkdir(recognition_abs_path)

        # Initialize tableWidget
        # TODO: 이 위젯이 언제 초기화되어야하는지 생각해봐야함
        # tableWidget에 있는 모든 row 제거
        while tableWidget.rowCount() > 0:
            tableWidget.removeRow(0);
        # tableWidget.setRowCount(len(selected_files))
        tableWidget.setColumnCount(6)
        tableWidget.setHorizontalHeaderLabels(["문제", "답안 이미지", "인식 결과", "정답", "채점 결과", "결과 수정"])
        QApplication.processEvents()
        
        all_imgs_doc = {}
        all_imgs_count = 0
        # detection되어서 나눠지는 전체 이미지 개수 구하고 table의 열 개수 설정
        for i in range(len(selected_files)):
            file_src_dir = selected_files[i]
            file_name = os.path.split(file_src_dir)[-1].strip()
            sheet_name = os.path.splitext(file_name)[0].strip()

            # 현재 요소가 file이 아닌 경우
            if not os.path.isfile(file_src_dir):
                # tableWidget.setItem(i, 5, QTableWidgetItem("파일을 찾을 수 없습니다."))
                QApplication.processEvents()
                continue
            
            # 파일이 맞는 경우
            _, ext = os.path.splitext(file_name)
            if ext == ".png":
                # 파일 업로드
                shutil.copy(file_src_dir, dst_abs_path)

                # get img_diff
                original_img_list = os.listdir(original_image_abs_path)
                base_name = ""
                for s in sheet_name.split("_")[:-1]:
                    base_name += (s + "_")
                base_name = base_name[:-1]
                print(base_name + ".png")
                diff_successful = False
                if base_name + ".png" in original_img_list:
                    img_diff.get_diff_img(file_src_dir, os.path.join(original_image_abs_path, base_name + ".png"), diff_image_abs_path)
                    diff_successful = True

                detect_target_img_dir = os.path.join(dst_abs_path, file_name)
                if diff_successful:
                    detect_target_img_dir = os.path.join(diff_image_abs_path, file_name)

                # 필기체 detection
                detected_img_dirs = text_detection_contour.detect2(detect_target_img_dir, detection_abs_path)
                all_imgs_doc[sheet_name] = detected_img_dirs
                all_imgs_count += len(detected_img_dirs)

        tableWidget.setRowCount(all_imgs_count)
        QApplication.processEvents()

        # DEBUG
        print(all_imgs_doc)

        # 채점 진행
        i = 0 # row count
        for sheet_name, img_dirs in all_imgs_doc.items():
            for j in range(len(img_dirs)):
                img_dir = img_dirs[j]
                tableWidget.setItem(i+j, 0, QTableWidgetItem(sheet_name))
                QApplication.processEvents()
                # load image
                img = QPixmap(img_dir)
                img = img.scaledToHeight(64)
                img_label = QLabel()
                img_label.setPixmap(img)
                tableWidget.setCellWidget(i+j, 1, img_label)
                tableWidget.setRowHeight(i+j, 64)
                tableWidget.setColumnWidth(1, 150)
                QApplication.processEvents()

                # 인식 폴더에 detection된 이미지 모두 복사해두기
                shutil.copy(img_dir, recognition_abs_path)
            QApplication.processEvents()

            # 정답 가져오기
            base_name = ""
            for s in sheet_name.split("_")[:-1]:
                base_name += (s + "_")
            base_name = base_name[:-1]
            print(base_name)
            correct_answer_abs_dir = os.path.join(correct_answer_abs_path, base_name + ".txt")
            correct_answers = None
            if os.path.isfile(correct_answer_abs_dir):
                with open(correct_answer_abs_dir, "r", encoding="utf-8") as caf:
                    correct_answers = caf.readlines()
                    print(correct_answers)
            if correct_answers != None:
                correct_answers = [s.strip() for s in correct_answers]
                for j in range(len(correct_answers)):
                    tableWidget.setItem(i+j, 3, QTableWidgetItem(correct_answers[j].strip()))
                    QApplication.processEvents()

            # 필기체 인식
            grade_result = {}
            recognition_result = run_recognition(recognition_abs_path)
            for j in range(len(img_dirs)):
                img_dir = img_dirs[j]
                tableWidget.setItem(i+j, 2, QTableWidgetItem(recognition_result[j]))
                if os.path.isfile(os.path.join(recognition_abs_path, os.path.split(img_dir)[-1])):
                    os.remove(os.path.join(recognition_abs_path, os.path.split(img_dir)[-1]))
                # 정답비교
                if correct_answers != None:
                    if not sheet_name in grade_result:
                        grade_result[sheet_name] = []
                    try:
                        current_grade_result = str(recognition_result[j]).__eq__(str(correct_answers[j]))
                        grade_result[sheet_name].append(current_grade_result)
                        tableWidget.setItem(i+j, 4, QTableWidgetItem(str(current_grade_result)))
                        if current_grade_result:
                            tableWidget.item(i+j, 4).setBackgroundColor(true_color)
                        else:
                            tableWidget.item(i+j, 4).setBackgroundColor(false_color)
                        print(current_grade_result)
                        # 결과인식 수정버튼
                        changeButton = ChangeButton("결과 수정", sheet_name, j, current_grade_result, i+j, tableWidget)
                        tableWidget.setCellWidget(i+j, 5, changeButton)
                    except Exception as e: # out of index
                        print(e)
                        continue
                QApplication.processEvents()

            # 채점결과 저장
            if correct_answers != None:
                with open(os.path.join(grade_result_abs_path, sheet_name) + ".txt", "w", encoding="utf-8") as file:
                    for j in range(len(img_dirs)):
                        file.write(
                            str(sheet_name) + "|" +
                            str(img_dirs[j]) + "|" + 
                            str(recognition_result[j]) + "|" + 
                            str(correct_answers[j]) + "|" + 
                            str(grade_result[sheet_name][j]) + "\n"
                        )
            
            # DEBUG
            print(grade_result)

            # 루프 마지막에 len(img_dirs)만큼 증가
            i += len(img_dirs)
            # 인식 폴더 비우기
            something_wrong_imgs = os.listdir(recognition_abs_path)
            for x in something_wrong_imgs:
                os.remove(x)

    def get_grade_results(self, tableWidget):
        # Initialize tableWidget
        # TODO: 이 위젯이 언제 초기화되어야하는지 생각해봐야함
        # tableWidget에 있는 모든 row 제거
        while tableWidget.rowCount() > 0:
            tableWidget.removeRow(0)
        tableWidget.setColumnCount(6)
        tableWidget.setHorizontalHeaderLabels(["문제", "답안 이미지", "인식 결과", "정답", "채점 결과", "결과 수정"])
        QApplication.processEvents()

        # 채점 결과 확인에 필요한 폴더들 확인
        data_abs_path = os.path.join(self.cwd, "data")
        grade_result_abs_path = os.path.join(data_abs_path, "grade_results")
        answer_sheet_abs_path = os.path.join(data_abs_path, "answer_sheets")
        detection_abs_path = os.path.join(answer_sheet_abs_path, "detection_result")
        if not os.path.exists(data_abs_path) or not os.path.exists(grade_result_abs_path) or not os.path.exists(answer_sheet_abs_path) or not os.path.exists(detection_abs_path):
            self.show_messageBox(MessageBoxType.CRITICAL, text="채점 결과가 없습니다.", title="Info")
        
        # 결과 가져오기
        grade_results = os.listdir(grade_result_abs_path)
        result_count = 0
        row_count = 0
        for res in grade_results:
            with open(os.path.join(grade_result_abs_path, res), "r", encoding="utf-8") as file:
                data = file.readlines()
            data = [x.strip().split("|") for x in data]
            tableWidget.setRowCount(row_count + len(data))
            for i in range(len(data)):
                for j in range(len(data[i])):
                    if j == 1:
                        # load image
                        img = QPixmap(data[i][j])
                        img = img.scaledToHeight(64)
                        img_label = QLabel()
                        img_label.setPixmap(img)
                        tableWidget.setCellWidget(row_count + i, 1, img_label)
                        tableWidget.setRowHeight(row_count + i, 64)
                        tableWidget.setColumnWidth(1, 150)
                    else:
                        # default
                        tableWidget.setItem(row_count + i, j, QTableWidgetItem(data[i][j]))
                        if j == 4:
                            if data[i][j].__eq__("True"):
                                tableWidget.item(row_count+i, j).setBackgroundColor(true_color)
                            else:
                                tableWidget.item(row_count+i, j).setBackgroundColor(false_color)
                changeButton = ChangeButton("결과 수정", data[i][0], i, data[i][4], row_count + i, tableWidget)
                tableWidget.setCellWidget(row_count + i, 5, changeButton)
                print("Button created successfully")
                QApplication.processEvents()
            row_count += len(data)


class ChangeButton(QPushButton):
    def __init__(self, text, sheet_name, index_of_img, grade_result, row, tableWidget):
        super().__init__(text)
        self.setText(text)
        self.sheet_name = sheet_name
        self.index_of_img = index_of_img
        self.grade_result = grade_result
        self.row = row
        self.tableWidget = tableWidget
        self.clicked.connect(self.change_grade_result)
    
    def __del__(self):
        print("button deleted")

    def change_grade_result(self):
        print("clicked", self.sheet_name, self.index_of_img, self.grade_result)
        data_abs_path = os.path.join(os.getcwd(), "data")
        grade_result_abs_path = os.path.join(data_abs_path, "grade_results")
        grade_result_file = os.path.join(grade_result_abs_path, self.sheet_name) + ".txt"
        grade_result_data = []
        if os.path.isfile(grade_result_file):
            with open(grade_result_file, "r", encoding="utf-8") as file:
                grade_result_data = file.readlines()
            grade_result_data = [x.strip().split("|") for x in grade_result_data]
            print(grade_result_data)
            print(grade_result_data[self.index_of_img][-1])
            print(type(grade_result_data[self.index_of_img][-1]))
            true_false = True if str(grade_result_data[self.index_of_img][-1]).__eq__("True") else False
            print(true_false, not true_false)
            grade_result_data[self.index_of_img][-1] = str(not true_false)
            with open(grade_result_file, "w", encoding="utf-8") as file:
                for line in grade_result_data:
                    line_to_write = ""
                    for content in line:
                        line_to_write += (content + "|")
                    file.write(line_to_write[:-1] + "\n")
            self.tableWidget.setItem(self.row, 4, QTableWidgetItem(str(not true_false)))
            if not true_false:
                self.tableWidget.item(self.row, 4).setBackgroundColor(true_color)
            else:
                self.tableWidget.item(self.row, 4).setBackgroundColor(false_color)
        else:
            print("Grade result file not found")


if __name__ == "__main__":
    # Main
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
