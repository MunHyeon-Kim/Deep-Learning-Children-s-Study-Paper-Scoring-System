# -*- coding: utf-8 -*-
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

# from utils import CTCLabelConverter, AttnLabelConverter
# from dataset import RawDataset, AlignCollate
# from model import Model

from text_recogition_naverAi.recognition.utils import CTCLabelConverter, AttnLabelConverter
from text_recogition_naverAi.recognition.dataset import RawDataset, AlignCollate
from text_recogition_naverAi.recognition.model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_folder = ''

def levenshtein(s1, s2, debug=False):
    if len(s1) < len(s2):
        return levenshtein(s2, s1, debug)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]


def demo(opt):
    """ model configuration """
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
    
                # # Todo
                # # 1. extract labels from test images
                # with open(opt.label_test, 'r', encoding="UTF-8") as f:
                #     lines = f.readlines()
                #     for line in lines:
                #         info = line.split('.png\t')
                #         file_name = info[0] + '.png'
                #         label = info[1].strip()
                #         if file_name.split('/')[-1] == img_name.split('/')[-1]: break
                #
                # # 2. calculate CER
                # # CER: Ground Truth(img_name)를 OCR 출력(pred)로 변환하는데 필요한 최소 문자 수준 작업 수
                # # CER = 100 * [1 - (탈자개수 + 오자개수 + 첨자개수) / 원본글자수]
                # error_num = levenshtein(pred, label, debug=True) # 오자 + 탈자 + 첨자
                # cer = error_num / len(label)                     # 현재 텍스트에 대한 cer
                # total_len += len(label)                          # 전체 텍스트(ground truth)의 길이
                # total_err += error_num                              # 전체 텍스트의 오자+탈자+첨자 수
                # total_cer = total_err / total_len                   # 전체 텍스트에 대한 cer
                predlist.append(pred)
                print(f'{img_name:25s}\t{pred:15s}\t{confidence_score:0.4f}')
                log.write(f'{img_name:25s}\t{pred:15s}\t{confidence_score:0.4f}')

            log.close()

    return predlist

