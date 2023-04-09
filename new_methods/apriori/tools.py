import os.path
from itertools import permutations
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

num = 0


def generate_apriori(img_path, cam, store_path, minimum_support):
    img = cv2.imread(img_path)
    cam = cam.cpu().squeeze(0).detach().numpy()
    img_shape = img.shape
    # print(img.shape)

    if not os.path.exists(store_path): os.makedirs(store_path)

    df: pd.DataFrame = pd.DataFrame(
        columns=list(permutations(range(cam.shape[1]), 2)) + [(x, x) for x in range(cam.shape[1])])

    pre_processed = []

    for n in range(cam.shape[0]):
        pixel_coords = []
        for x, y in df.columns:
            if cam[n][x][y] > 0.02:
                pixel_coords.append((x, y))
        pre_processed.append(pixel_coords)

    te = TransactionEncoder()
    te_ary = te.fit(pre_processed).transform(pre_processed)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df, min_support=minimum_support, use_colnames=True)

    a = []
    b = []

    for y, x in zip(frequent_itemsets['support'], frequent_itemsets['itemsets']):
        for i in x:
            a.append(i)
            # b.append(y)

    img = np.zeros([cam.shape[1], cam.shape[2]])
    for i in range(cam.shape[1]):
        for j in range(cam.shape[2]):
            if (i, j) in a:
                img[i][j] = 1

    # TODO Save the Image
    # TODO heatmap
    img = cv2.resize(img, (img_shape[0], img_shape[1]))
    # heatmap = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    #
    # result = heatmap * 0.3 + img * 0.5
    #
    # result = np.uint8(255 - 1 * result)
    # print(result.shape)
    # cv2.imshow('map', result)
    global num
    # plt.imshow(img)
    # if minimum_support==0.4:
    #     store_path=store_path+'scam_'+str(num)
    # else:
    #     store_path=store_path+'dcam_'+str(num)
    #     num += 1
    cv2.imwrite(store_path + img_path.split('/')[-1], img)
    # plt.savefig(os.path.join(store_path, img_path.split('/')[-1]))
    # plt.show()
    return img


def generate_heatmap(img_path, map):
    img = cv2.imread(img_path)
    cam = map.cpu().squeeze(0).detach().numpy()
    cam_img = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    result = np.uint8(255 - 1 * result)
    print(result.shape)
