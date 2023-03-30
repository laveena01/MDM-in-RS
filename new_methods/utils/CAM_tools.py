
import cv2
import numpy as np
import xml.dom.minidom
from new_methods.utils.bndresult import bndresult

sj = 0.2
xj = 0.002


def deleteBBox(bbox_before, area):
    # delete some unresonable bbox
    #:param bbox_before:
    #:param area:
    #:return:
    pass


def generateCAM(img_path, cam, store_path=None, labels=None):
    """
    generate CAM use diffrent method
    :param img:
    :param cam:
    :param labels:
    :return: CAM
    """

    img = cv2.imread(img_path)
    cam = cam.cpu().squeeze(0).detach().numpy()
    img_shape = img.shape
    # print("SIZE b ",img_shape)
    # print(f"Size = {cam.shape}")
    if labels is not None:
        cam = cam[labels]
    else:
        cam = cam.sum(axis=0) #  row wise sum
    # print(f"Size = {cam.shape}")
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    # print(cam_img.shape)
    cam_img = cv2.resize(cam_img, (img_shape[1], img_shape[0]))
    # cv2.imshow('image', cam_img)
    # src_img = img.transpose(1, 2, 0)
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # src_img = std * src_img + mean
    # src_img = np.clip(src_img, 0, 1)
    # src_img = np.uint8(255 * src_img)
    # if labels is  None:

    heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)

    result = heatmap * 0.3 + img * 0.5

    result = np.uint8(255 - 1 * result)
    # print(result.shape)
    # cv2.imshow('map', result)
    # cv2.waitKey(3500)
    # #
     #if labels is None:
    #     cv2.imwrite(store_path+'/heatmap_3/'+img_path.split('/')[-1], result)
    # else:
    #     cv2.imwrite(store_path+'/heatmap_4/'+img_path.split('/')[-1], result)

    return cam_img


def generateBBox(cam, val):
    # use cam generate bbox
    #:param img: orgin image
    #:param cam: cam image
    #:return: list(bbox)

    if val:
        ret, thresh = cv2.threshold(cam.copy(), 0, 100,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hier = cv2.findContours(cam.astype(np.uint8), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
    res = []
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        res.append(bndresult(x, y, x + w, y + h))
    return res


def compute_iou(a, b, flag=True):
    # :param a: Boundingbox pre
    #:param b: Boundingbox gt
    # :return:

    S_rec1 = (a.x2 - a.x1) * (a.y2 - a.y1)
    S_rec2 = (b.x2 - b.x1) * (b.y2 - b.y1)
    if not flag:
        sum_area = min(S_rec1, S_rec2)
    else:
        sum_area = S_rec1 + S_rec2
    left_line = max(a.y1, b.y1)
    right_line = min(a.y2, b.y2)
    top_line = max(a.x1, b.x1)
    bottom_line = min(a.x2, b.x2)

    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        if sum_area == intersect:
            return 1
        return float(intersect) / (sum_area - intersect)


def comparePretoGt(pre, gt):
    # compute TP, FP, FN
    #:param pre: current image pre Box
    #:param gt: current image Gt Box
    #:return: [TP, Fp, FN]

    TP = 0
    bbox_pre0_5 = []
    for i in gt:
        for j in pre:
            iou = compute_iou(i, j)
            if iou >= 0.50:
                TP = TP + 1
                bbox_pre0_5.append(j)
                # break

    FN = len(gt) - TP
    FP = len(pre) - TP
    return (TP, FP, FN), bbox_pre0_5


def mergeCAM(Box3, Box4):
    # merge the result with layer3 and layer4
    # :param Box3: layer3
    #:param Box4: layer4
    #:return: list(bbox)

    res = []
    for c4 in Box4:
        flag = 0
        for c3 in Box3:
            area = compute_iou(c4, c3)
            if area >= 0.01 and area <= 1:
                if not c3 in res:
                    c3.categories = c4.categories
                    c3.confidence = c4.confidence
                    res.append(c3)
                flag = 1
                # break
        if flag == 0:
            res.append(c4)

    return res


def generateBBoxGT(str):
    # generate GTBBox
    #:param str:
    #:return:

    res = []
    DOMTree = xml.dom.minidom.parse(str)
    collection = DOMTree.documentElement
    filenamelist = collection.getElementsByTagName("filename")
    filename = filenamelist[0].childNodes[0].data

    objectlist = collection.getElementsByTagName("object")

    for objects in objectlist:

        namelist = objects.getElementsByTagName('name')

        objectname = namelist[0].childNodes[0].data

        bndbox = objects.getElementsByTagName('bndbox')

        for box in bndbox:
            x1_list = box.getElementsByTagName('xmin')
            x1 = int(x1_list[0].childNodes[0].data)
            y1_list = box.getElementsByTagName('ymin')
            y1 = int(y1_list[0].childNodes[0].data)
            x2_list = box.getElementsByTagName('xmax')
            x2 = int(x2_list[0].childNodes[0].data)
            y2_list = box.getElementsByTagName('ymax')
            y2 = int(y2_list[0].childNodes[0].data)
            # (y0, x0, y1, x1)
            res.append(bndresult(x1, y1, x2, y2))
    # print(res)
    return res


def drawRect(img, contours, color):
    for c in contours:
        cv2.rectangle(img, (c.x1, c.y1), (c.x2, c.y2), color, 2)
    return img


if __name__ == "__main__":
    print
    # img = cv2.imread('airplane_001.jpg')
    # cam = cv2.imread('test.jpg')

    # test generatCAM
    # pre = generateBBox(cam)
    # for c in pre:
    #     cv2.rectangle(img, (c.x1, c.y1), (c.x2, c.y2), (0, 255, 0), 2)
    # cv2.imshow('t', img)
    # cv2.waitKey(0)

    # test iou