# 正規化していないlandmarksを後から正規化する際に使う

import math
import csv
import os
import sys
import copy
import itertools
import time
import datetime
import numpy as np

def calc_landmark_list(img_width, img_height, landmarks):

    landmarks = landmarks[1:]

    # 画面上のランドマークの位置を算出
    for id, landmark in enumerate(landmarks):
        #print(id, landmark)
        # x軸
        if np.mod(id,2) == 0:
            landmarks[id] = min(int(landmark * img_width), img_width - 1)
        # y軸
        elif np.mod(id,2) == 1:
            landmarks[id] = min(int(landmark * img_height), img_height - 1)

    landmarks_reordered = [[landmarks[i], landmarks[i+1]] for i in range(0, len(landmarks), 2)]

    return landmarks_reordered


def pre_process_landmark(landmark_list):

    temp_landmark_list = copy.deepcopy(landmark_list)

    # 相対座標に変換
    base_x, base_y = 0, 0

    for index, landmark_list in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_list[0], landmark_list[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # 1次元リストに変換
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # 最初の2つは0なので削除
    temp_landmark_list = temp_landmark_list[2:]

    return temp_landmark_list



# csv保存
def write_csv_normalised(yubimoji_id, normalised_landmarks, starttime):

    starttime = starttime.strftime('%Y%m%d%H%M%S')

    csv_path = './point_history_normalised.csv'

    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([yubimoji_id, *normalised_landmarks])


with open('./point_history.csv', encoding='utf-8-sig') as f:

    starttime = datetime.datetime.now()

    landmarks_list = csv.reader(f)
    cnt =0 

    for landmarks in landmarks_list:

        for i in range(len(landmarks)):
            if i == 0:
                landmarks[i] = int(landmarks[i])
                yubimoji = landmarks[i]
            else:
                landmarks[i] = float(landmarks[i])

        landmarks = calc_landmark_list(640, 480, landmarks) # 20 * 2
        normalised_landmarks = pre_process_landmark(landmarks) # 40 *1

        print(normalised_landmarks)

        write_csv_normalised(yubimoji, normalised_landmarks, starttime)      
    print(cnt)