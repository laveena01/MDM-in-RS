
import csv

import numpy

from new_methods.expr.train import device
from new_methods.data.airplane_data import *
from CAM_tools import *
import os
from PIL import Image
import cv2
import pickle
from new_methods.utils.visualise import show


def test(model, store_path):
    model.to(device)
    model.eval()

    # load data
    test_data_size = 0
    corrects = 0.0
    str = r'D:\CODE_AND_RESULTS\data\test_airplane'

    TP = [0.0, 0.0, 0.0]
    FP = [0.0, 0.0, 0.0]
    FN = [0.0, 0.0, 0.0]

    # preBBox_num = 0
    # BBox_num = 0

    for img in os.listdir(str + '/img'):
        # for img in os.listdir(str + ):
        # inputs
        image = Image.open(str + '/img/' + img)
        # image = Image.open(str + '\\train\\positive img/' + img)
        inputs = test_data_transform(image)
        inputs = inputs.unsqueeze(0)
        inputs = inputs.to(device)

        # outputs
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        print("Hi")
        print(preds)

        layer_map3, layer_map4 = model.get_salience_maps()
        cam3 = generateCAM(str + '/img/' + img, layer_map3, store_path, None)
        cam4 = generateCAM(str + '/img/' + img, layer_map4, store_path, preds)
        # cam3a =
        # cam4a =
        # print(cam3)
        # cv2.imshow('image',cam3)
        # cv2.waitKey(0)
        # print(cam4)
        # cv2.imshow('image', cam4)
        # cv2.waitKey(0)
        bbox_pre3 = generateBBox(cam3)
        # bbox_pre3a =
        # ar1=numpy.array(bbox_pre3[0])
        # print(bbox_pre3[0])
        # cv2.imshow('image', ar1)
        # cv2.waitKey(0)
        bbox_pre4 = generateBBox(cam4)
        # ar2 = numpy.array(bbox_pre4)
        # cv2.imshow('image', ar2)
        # cv2.waitKey(0)
        bbox_gt = generateBBoxGT(str + '/annotations/' + os.path.splitext(img)[0] + '.xml')
        bbox_pre34 = mergeCAM(bbox_pre3, bbox_pre4)
        # cv2.imshow('image', bbox_pre34)
        # cv2.waitKey(0)
        # ar3 = numpy.array(bbox_pre34)
        # print(type(ar3))
        # print(bbox_pre34)

        value_temp, bbox_pre0_5_3 = comparePretoGt(bbox_pre3, bbox_gt)
        TP[0] = TP[0] + value_temp[0]
        FP[0] = FP[0] + value_temp[1]
        FN[0] = FN[0] + value_temp[2]

        value_temp, bbox_pre0_5_4 = comparePretoGt(bbox_pre4, bbox_gt)
        TP[1] = TP[1] + value_temp[0]
        FP[1] = FP[1] + value_temp[1]
        FN[1] = FN[1] + value_temp[2]

        value_temp, bbox_pre0_5_34 = comparePretoGt(bbox_pre34, bbox_gt)
        TP[2] = TP[2] + value_temp[0]
        FP[2] = FP[2] + value_temp[1]
        FN[2] = FN[2] + value_temp[2]

        # BBox_num = BBox_num + len(bbox_gt)
        # preBBox_num = preBBox_num + len(bbox_pre34)

        # visulize result
        rr = [bbox_pre3, bbox_pre4, bbox_pre34]

        i = 0
        for k in ['/3/', '/4/', '/3+4/']:
            vi_img = cv2.imread(str + '/img/' + img)
            vi_img = drawRect(vi_img, bbox_gt, (0, 255, 0))
            vi_img = drawRect(vi_img, rr[i], (0, 0, 255))
            cv2.imwrite(store_path + k + img, vi_img)
            i = i + 1
        #
        cv2.imshow('t', vi_img)
        cv2.waitKey(5500)

        corrects += torch.sum(preds == 1)
        test_data_size = test_data_size + 1

    print('{:.2f}'.format(corrects / test_data_size))

    for i in range(3):
        print('TP: {:.0f} FP: {:.0f} FN: {:.0f}'.format(TP[i], FP[i], FN[i]))
        print('Precison : {:.4f}  Recall : {:.4f}'.format(TP[i] / (TP[i] + FP[i]), TP[i] / (TP[i] + FN[i])))
        print()

    # print 'BBox_num: {:.0f} preBBox_num: {:.0f}'.format(BBox_num, preBBox_num)
    return [corrects / test_data_size, TP, FP, FN]


if __name__ == "__main__":
    store_path = r'D:\CODE_AND_RESULTS\new_methods\utils\no_DA'
    path = r'D:\CODE_AND_RESULTS\new_methods\utils\model_trained\tparams_single2_'
    select_model = 'DA_PAM'
    #select_model = 'CAM'
    methods = ['pam', 'cam']
    K = [16]  # correct
    # COS = [0.01, 0.02, 0.1, 0.5, 0.7] #correct
    COS = [0.05]
    CA = []
    TA = []
    FA = []
    PA = []

    cam = False
    pam = False

    file = os.path.join(store_path, 'result.csv')

    with open(file, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, dialect='excel')
        spamwriter.writerow(['method', 'K', 'COS', 'TP', 'FP', 'FN', 'Precison', 'Recall', 'cls_res'])
        for method in methods:
            if method == 'all':
                cam = True
                pam = True
            if method == 'pam':
                pam = True
            if method == 'cam':
                cam = True
            count = 0
            for k in K:
                for cos in COS:

                    #model_path = path + select_model + '_' + str(cos) + '_' + str(k) + '_0_' + method+'.pkl'  #correct
                    model_path = path + select_model + '_' + str(cos) + '_' + str(k) + '.pkl'  # trial
                    # load model
                    from new_methods.model.resnet_DA_PAM import model
                    #from new_methods.model.resnet import model

                    #model_ft = model(pretrained=True, num_classes=2)
                    model_ft = model(pretrained=True, num_classes=2, num_maps=k, cos_alpha=cos, pam=pam, cam=cam)
                    model_ft = model_ft.to(device)
                    model_ft.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

                    final_path = os.path.join(store_path, method, str(k) + '_' + str(cos))
                    for t in ['3', '4', '3+4', 'heatmap_3', 'heatmap_4']:
                        temp = os.path.join(final_path, t)

                        if not os.path.exists(temp):
                            os.makedirs(temp)

                    res = test(model_ft, final_path)
                    for i in [2]:
                        spamwriter.writerow([str(method), str(k), str(cos), res[1][i], res[2][i], res[3][i],
                                             res[1][i] / (res[1][i] + res[2][i]), res[1][i] / (res[1][i] + res[3][i]),
                                             res[0]])
                        pass
                    # count = count + 1
                    # CA.append(count)
                    # PA.append(res[0])
                    # TA.append(res[1])
                    # FA.append(res[2])

            # show(CA, TA, FA, PA)
            # CA = []
            # PA = []
            # TA = []
            # FA = []
            pam = False
            cam = False