# Outputとなるクラス数: 87
# 清音46 + 濁音・半濁音・拗音・長音・促音30 + 数字10 + コマンド1 (入力削除用) 

import math
import csv
import os
import sys
import cv2
import copy
import pprint
import itertools
import time
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import datetime
import tensorflow as tf
from model import KeyPointClassifier

def main(mode, yubimoji_id=None):
    # mode: 0 -> 録画モード, 1 -> 予測モード

    cnt = 0
    Textcnt = ''
    Textcnt_list = [' 正面',' 左上',' 上',' 右上',' 左',' 正面',' 右',' 左下',' 下',' 右下']

    # yubimoji_idが存在しないID、もしくは空の場合、停止
    if mode == 1 and yubimoji_id != None:
        if yubimoji_id < 0 or yubimoji_id > 89:
            sys.exit()

    # 実行中のファイルのパスをカレントディレクトリに設定
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # MediaPipeモデルのロード
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,                # 最大検出数
        min_detection_confidence=0.8,   # 検出信頼度
        min_tracking_confidence=0.8     # 追跡信頼度
    )

    # 予測モデルのロード
    if mode == 1:
        keypoint_classifier = KeyPointClassifier("model/keypoint_classifier/keypoint_classifier.tflite")

    # カメラキャプチャ設定
    video_capture = cv2.VideoCapture(0) #内臓カメラ
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    # 指文字のラベル
    with open('setting/hand_keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        yubimoji_labels = csv.reader(f)
        yubimoji_labels = [row[0] for row in yubimoji_labels]

    # (録画のみ) 指文字登録用変数
    starttime = datetime.datetime.now()

    if video_capture.isOpened():

        while True:

            # 3秒経ったらプログラム終了
            sec_dif = datetime.datetime.now() - starttime
            sec_dif = sec_dif.total_seconds()  
            if mode == 0 and sec_dif > 3:
                time.sleep(1)
                starttime = datetime.datetime.now() 
                cnt += 1

            if cnt > 9:
                break

            # キー入力(ESC:プログラム終了)
            key = cv2.waitKey(1)
            if key == 27: break

            # カメラから画像取得
            success, img = video_capture.read()
            #frame_width, frame_height = img.shape[1], img.shape[0]
            if not success: break

            # 画像を左右反転
            img = cv2.flip(img, 1)

            # 検出処理の実行
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:

                for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                    # 手の左右判定 / 今回の副テーマ研究では左手の対応はしない
                    if handedness.classification[0].label[0:] == 'Right':

                        # ランドマークの画像上の位置を算出する関数
                        # 21ランドマークのx,y座標を算出 (z座標が必要になる場合は関数内で調整)
                        landmark_list = calc_landmark_list(img, landmarks)

                        # 中指第一関節 - 手首の距離計算 (Priyaら (2023))
                        palmsize = calc_palmsize(landmarks)

                        # 画面表示: 中指第一関節 - 手首の距離の表示
                        #cv2.putText(img, text=str(palmsize), org=(200,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(64, 0, 0), thickness=1, lineType=cv2.LINE_AA)

                        normalised_landmarks = pre_process_landmark(landmark_list, palmsize)

                    ##### 編集中
                        ## ここに相対座標・正規化座標の計算を追加する
                        if mode == 1:
                            point_history_list = []

                            # 各ランドマークの座標を取得
                            for i in range(len(landmarks.landmark)):
                                lm = landmarks.landmark[i]
                                point_history_list.append(lm.x)
                                point_history_list.append(lm.y)
                                
                            point_history_array = np.array(point_history_list)
                            point_history_list = point_history_array.reshape((1, 42))

                    ##### 編集中

                        # 文字ラベルの予測
                        if mode == 1:
                            #yubimoji_id = 0
                            yubimoji_id = keypoint_classifier(point_history_list)

                        # 画面表示: 文字表示
                        Textcnt = Textcnt_list[cnt] if mode == 0 and yubimoji_id != None else ''
                        img = putText_japanese(img, yubimoji_labels[yubimoji_id] + Textcnt)
                        
                        # 画面表示: ランドマーク間の線を表示
                        img = showLandmarkLines(img, landmarks)

                        # 録画のみ: 検出情報をcsv出力
                        if mode == 0:
                            write_csv(yubimoji_id, landmarks, starttime)
                            write_csv_normalised(yubimoji_id, normalised_landmarks, starttime)      

            # 画像の表示
            cv2.imshow("MediaPipe Hands", img)

    # リソースの解放
    video_capture.release()
    hands.close()
    cv2.destroyAllWindows()


def calc_landmark_list(img, landmarks):

    img_width, img_height = img.shape[1], img.shape[0]
    landmark_list = []

    # 画面上のランドマークの位置を算出
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * img_width), img_width - 1)
        landmark_y = min(int(landmark.y * img_height), img_height - 1)
        landmark_list.append([landmark_x, landmark_y])

    return landmark_list


def pre_process_landmark(landmark_list, palmsize):

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

    # 正規化
    def normalize_(n): return n / palmsize
    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    # 最初の二つは0なので削除
    del temp_landmark_list[0:2]

    return temp_landmark_list


# 中指第一関節 - 手首の距離計算
def calc_palmsize(landmarks):

    # 手首の座標を取得
    lm_0 = landmarks.landmark[0]

    # 中指第一関節の座標を取得
    lm_9 = landmarks.landmark[9]

    # 中指第一関節 - 手首の距離を計算
    distance_x = lm_0.x - lm_9.x
    distance_y = lm_0.y - lm_9.y
    distance_z = lm_0.z - lm_9.z # 角度の修正計算ができれば、ここは不要

    # 中指第一関節 - 手首の距離を代入
    palmsize = math.sqrt(distance_x ** 2 + distance_y ** 2 + distance_z ** 2)

    return palmsize


# 日本語表示
def putText_japanese(img, text):

    #Notoフォント
    font = ImageFont.truetype('/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc', size=20)

    #imgをndarrayからPILに変換
    img_pil = Image.fromarray(img)

    #drawインスタンス生成
    draw = ImageDraw.Draw(img_pil)

    #テキスト描画
    draw.text(
        xy=(20, 20),
        text=text, 
        font=font, 
        fill=(0, 0, 0)
    )

    #PILからndarrayに変換して返す
    return np.array(img_pil)


# ランドマーク間の線を表示
def showLandmarkLines(img, landmarks):

    # 各ランドマークの座標はlandmarks.landmark[0]~[20].x, .y, .zに格納されている
    img_h, img_w, _ = img.shape

    # Landmark間の線
    landmark_line_ids = [ 
        # 手のひら
        (0, 1),     # 手首 - 親指第一関節
        (1, 5),     # 親指第一関節 - 人差し指第一関節
        (5, 9),     # 人差し指第一関節 - 中指第一関節
        (9, 13),    # 中指第一関節 - 薬指第一関節
        (13, 17),   # 薬指第一関節 - 小指第一関節
        (17, 0),    # 小指第一関節 - 手首
        (0,9),     # 中指第一関節 - 手首　Priya論文

        # 指
        (1, 2),     # 親指第一関節 - 親指第二関節
        (2, 3),     # 親指第二関節 - 親指第三関節
        (3, 4),     # 親指第三関節 - 親指先端
        (5, 6),     # 人差し指第一関節 - 人差し指第二関節
        (6, 7),     # 人差し指第二関節 - 人差し指第三関節
        (7, 8),     # 人差し指第三関節 - 人差し指先端
        (9, 10),    # 中指第一関節 - 中指第二関節
        (10, 11),   # 中指第二関節 - 中指第三関節
        (11, 12),   # 中指第三関節 - 中指先端
        (13, 14),   # 薬指第一関節 - 薬指第二関節
        (14, 15),   # 薬指第二関節 - 薬指第三関節
        (15, 16),   # 薬指第三関節 - 薬指先端
        (17, 18),   # 小指第一関節 - 小指第二関節
        (18, 19),   # 小指第二関節 - 小指第三関節
        (19, 20),   # 小指第三関節 - 小指先端
    ]

    # landmarkの繋がりをlineで表示
    for line_id in landmark_line_ids:
        # 1点目座標取得
        lm = landmarks.landmark[line_id[0]]
        lm_pos1 = (int(lm.x * img_w), int(lm.y * img_h))
        # 2点目座標取得
        lm = landmarks.landmark[line_id[1]]
        lm_pos2 = (int(lm.x * img_w), int(lm.y * img_h))
        # line描画
        cv2.line(img, lm_pos1, lm_pos2, (128, 0, 0), 1)

    # landmarkをcircleで表示
    z_list = [lm.z for lm in landmarks.landmark]
    z_min = min(z_list)
    z_max = max(z_list)
    for lm in landmarks.landmark:
        lm_pos = (int(lm.x * img_w), int(lm.y * img_h))
        lm_z = int((lm.z - z_min) / (z_max - z_min) * 255)
        cv2.circle(
            img,
            radius=3, 
            center=lm_pos, 
            color=(255, lm_z, lm_z), 
            thickness=-1
        )

    return img


# csv保存
def write_csv(yubimoji_id, landmarks, starttime):

    starttime = starttime.strftime('%Y%m%d%H%M%S')

    point_history_list = []

    # 各ランドマークの座標を取得
    for i in range(len(landmarks.landmark)):
        lm = landmarks.landmark[i]
        point_history_list.append(lm.x)
        point_history_list.append(lm.y)

    csv_path = './point_history_{0}_{1}.csv'.format(str(yubimoji_id).zfill(2),starttime)

    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([yubimoji_id, *point_history_list])


# csv保存
def write_csv_normalised(yubimoji_id, normalised_landmarks, starttime):

    starttime = starttime.strftime('%Y%m%d%H%M%S')

    point_history_list = normalised_landmarks

    csv_path = './point_history_normalised_{0}_{1}.csv'.format(str(yubimoji_id).zfill(2),starttime)

    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([yubimoji_id, *point_history_list])


if __name__ == "__main__": 

    '''
    0: あ	1: い	2: う	3: え	4: お
    5: か	6: き	7: く	8: け	9: こ	
    10: さ	11: し	12: す	13: せ	14: そ
    15: た	16: ち	17: つ	18: て	19: と	
    20: な	21: に	22: ぬ	23: ね	24: の
    25: は	26: ひ	27: ふ	28: へ	29: ほ	
    30: ま	31: み	32: む	33: め	34: も
    35: や	36: ゆ	37: よ	
    38: ら	39: り	40: る	41: れ	42: ろ	
    43: わ	44: を	45: ん	
    46: が	47: ぎ	48: ぐ	49: げ	50: ご	
    51: ざ	52: じ	53: ず	54: ぜ	55: ぞ
    56: だ	57: ぢ	58: づ	59: で	60: ど	
    61: ば	62: び	63: ぶ	64: べ	65: ぼ
    66: ぱ	67: ぴ	68: ぷ	69: ぺ	70: ぽ
    71: っ  72: ゃ  73: ゅ  74 :ょ  75:ー
    '''
    mode = 1
    #char = 70
    # 50cm

    #main(mode,char)
    main(mode)