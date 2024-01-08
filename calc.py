# 計算用の関数をまとめたファイル

import math
import copy
import itertools

# 手掌長 (手首と中指第一関節間の距離) の取得 (Priyaら (2023))
def palmLength(landmarks):

    # 手首の座標を取得
    lm_0 = landmarks.landmark[0]

    # 中指第一関節の座標を取得
    lm_9 = landmarks.landmark[9]

    # 中指第一関節 - 手首の距離を計算
    distance_x = lm_0.x - lm_9.x
    distance_y = lm_0.y - lm_9.y
    distance_z = lm_0.z - lm_9.z # 角度の修正計算ができれば、ここは不要

    # 中指第一関節 - 手首の距離を代入
    palmLength = math.sqrt(distance_x ** 2 + distance_y ** 2 + distance_z ** 2)

    return palmLength


# ランドマークの画像上の相対座標 (x,y座標) を算出する関数 (z座標が必要になる場合は関数内で調整)
def lmRelativeLoc(frame, landmarks):

    img_width, img_height = frame.shape[1], frame.shape[0]

    landmark_list = []

    # 画面上のランドマークの位置を算出
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * img_width), img_width - 1)
        landmark_y = min(int(landmark.y * img_height), img_height - 1)
        landmark_list.append([landmark_x, landmark_y])

    return landmark_list


# 手掌長で正規化
# 理由: 学習データに無い位置でジェスチャーをした場合、認識が上手くいかないため、一番最初の座標をもとに相対座標を取得する
def Normalisation(landmark_list, palmLength=None):

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

    # 最初の二行は0なので削除
    del temp_landmark_list[0:2]

    # 手掌長の入力があった場合、手掌長で正規化
    if palmLength != None:
        def normalize_(n): return n / palmLength

    # 正規化
    else: 
        max_value = max(list(map(abs, temp_landmark_list)))
        def normalize_(n): return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list