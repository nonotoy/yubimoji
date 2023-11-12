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
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import datetime
import tensorflow as tf
from model import KeyPointClassifier


def mainEstimate():

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
    keypoint_classifier = KeyPointClassifier("model/keypoint_classifier/keypoint_classifier.tflite")

    # カメラキャプチャ設定
    video_capture = cv2.VideoCapture(0) #内臓カメラ
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    # 指文字のラベル
    yubimoji_labels = fetchYubimojiLabels()

    if video_capture.isOpened():

        while True:

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

                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                    # 手の左右判定 / 今回の副テーマ研究では左手の対応はしない
                    if handedness.classification[0].label[0:] == 'Right':

                        # Landmark間の線を表示
                        img = showLandmarkLines(img, hand_landmarks)

                        # 中指第一関節 - 手首の距離の取得・表示 (Priyaら (2023))
                        img, distance_wrist2MFjoint = calc_distance_wrist2MFjoint(img, hand_landmarks)


                    ##### 編集中
                        ## ここに相対座標・正規化座標の計算を追加する

                        # ランドマークの画像上の位置を算出する関数
                        # 21ランドマークのx,y座標を算出 (z座標が必要になる場合は関数内で調整)
                        landmark_list = calc_landmark_list(img, hand_landmarks)

                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        '''
                        [0.0,
                        0.0,
                        -0.3274647887323944,
                        -0.1936619718309859,
                        -0.5598591549295775,
                        -0.426056338028169,
                        -0.7112676056338029,
                        -0.5563380281690141,
                        -0.8133802816901409,
                        -0.7288732394366197, #10

                        -0.3485915492957746,
                        -0.75,
                        -0.5422535211267606,
                        -0.9859154929577465,
                        -0.7077464788732394,
                        -0.8450704225352113,
                        -0.795774647887324,
                        -0.6725352112676056,
                        -0.24295774647887325,
                        -0.7676056338028169, #20

                        -0.5211267605633803,
                        -1.0,
                        -0.721830985915493,
                        -0.795774647887324,
                        -0.7922535211267606,
                        -0.5950704225352113,
                        -0.13028169014084506,
                        -0.7640845070422535,
                        -0.41901408450704225,
                        -0.9929577464788732, #30

                        -0.647887323943662,
                        -0.7887323943661971,
                        -0.7429577464788732,
                        -0.5845070422535211,
                        -0.014084507042253521,
                        -0.7359154929577465,
                        -0.2640845070422535,
                        -0.9366197183098591,
                        -0.4859154929577465,
                        -0.8661971830985915, #40

                        -0.6267605633802817,
                        -0.75] #42
                        '''
                        point_history_list = []

                        # 各ランドマークの座標を取得
                        for i in range(len(hand_landmarks.landmark)):
                            lm = hand_landmarks.landmark[i]
                            point_history_list.append(lm.x)
                            point_history_list.append(lm.y)

                        #point_history_list.append(distance_wrist2MFjoint)
                        #print(len(point_history_list))

                        pre_processed_landmark_list = point_history_list
                        # 文字ラベルの判定
                        yubimoji_id = keypoint_classifier(pre_processed_landmark_list)   #######エラー出る

                    ##### 編集中

                        # 予想した文字を表示
                        if yubimoji_id != None:
                            img = putText_japanese(img, yubimoji_labels[yubimoji_id])

            # 画像の表示
            cv2.imshow("MediaPipe Hands", img)

    # リソースの解放
    video_capture.release()
    hands.close()
    cv2.destroyAllWindows()



def mainRecord(yubimoji_id):

    # 実行中のファイルのパスをカレントディレクトリに設定
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # MediaPipeモデルのロード
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,                # 最大検出数
        min_detection_confidence=0.8,   # 検出信頼度
        min_tracking_confidence=0.8     # 追跡信頼度
    )

    # カメラキャプチャ設定
    video_capture = cv2.VideoCapture(0) #内臓カメラ
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    # 指文字のラベル
    yubimoji_labels = fetchYubimojiLabels()

    # 指文字登録用変数
    starttime = datetime.datetime.now()
    #yubimoji_id = gesture_id_start # 初期指文字ID
    gesture_id_end = yubimoji_id
    
    if video_capture.isOpened():

        while True:

            # 3秒ごとに次の指文字へ
            ts = datetime.datetime.now() - starttime
            if ts.seconds > 3:
                starttime = datetime.datetime.now()
                yubimoji_id += 1
            
            # gesture_idが存在しないID、もしくは引数で指定した範囲外のIDになった場合、停止
            if yubimoji_id < 0 or (yubimoji_id > 89 and yubimoji_id != 99) or yubimoji_id > gesture_id_end:
                sys.exit()

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

                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                    # 手の左右判定 / 今回の副テーマ研究では左手の対応はしない
                    if handedness.classification[0].label[0:] == 'Right':

                        # 画面表示: ランドマーク間の線
                        img = showLandmarkLines(img, hand_landmarks)

                        # 画面表示: 中指第一関節 - 手首の距離 (Priyaら (2023))
                        img, distance_wrist2MFjoint = calc_distance_wrist2MFjoint(img, hand_landmarks)

                        # 画面表示: 録画中の文字
                        if yubimoji_id != None:
                            img = putText_japanese(img, yubimoji_labels[yubimoji_id])

                        # 検出情報をcsv出力
                        write_csv(yubimoji_id, hand_landmarks, distance_wrist2MFjoint, starttime)
                    
            # 画像の表示
            cv2.imshow("MediaPipe Hands", img)

    # リソースの解放
    video_capture.release()
    hands.close()
    cv2.destroyAllWindows()


# 指文字のラベル
def fetchYubimojiLabels():

    with open('setting/hand_keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        yubimoji_labels = csv.reader(f)
        yubimoji_labels = [row[0] for row in yubimoji_labels]

    return yubimoji_labels


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


# csv保存
def write_csv(yubimoji_id, hand_landmarks, distance_wrist2MFjoint, starttime):

    starttime = starttime.strftime('%Y%m%d%H%M%S')

    point_history_list = []

    # 各ランドマークの座標を取得
    for i in range(len(hand_landmarks.landmark)):
        lm = hand_landmarks.landmark[i]
        point_history_list.append(lm.x)
        point_history_list.append(lm.y)
        # point_history_list.append(lm.z)

    point_history_list.append(distance_wrist2MFjoint)

    csv_path = './point_history_{0}_{1}.csv'.format(str(yubimoji_id).zfill(2),starttime)

    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        #writer.writerow([yubimoji_id, width, height, *point_history_list])
        writer.writerow([yubimoji_id, *point_history_list])


def showLandmarkLines(img, hand_landmarks):

    # 各ランドマークの座標はhand_landmarks.landmark[0]~[20].x, .y, .zに格納されている
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
        lm = hand_landmarks.landmark[line_id[0]]
        lm_pos1 = (int(lm.x * img_w), int(lm.y * img_h))
        # 2点目座標取得
        lm = hand_landmarks.landmark[line_id[1]]
        lm_pos2 = (int(lm.x * img_w), int(lm.y * img_h))
        # line描画
        cv2.line(img, lm_pos1, lm_pos2, (128, 0, 0), 1)

    # landmarkをcircleで表示
    z_list = [lm.z for lm in hand_landmarks.landmark]
    z_min = min(z_list)
    z_max = max(z_list)
    for lm in hand_landmarks.landmark:
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


def calc_landmark_list(img, landmarks):

    img_width, img_height = img.shape[1], img.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * img_width), img_width - 1)
        landmark_y = min(int(landmark.y * img_height), img_height - 1)
        #landmark_z = landmark.z
        #landmark_point.append([landmark_x, landmark_y, landmark_z])
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):

    temp_landmark_list = copy.deepcopy(landmark_list)

    #print(temp_landmark_list) #21

    # 相対座標に変換
    # base_x, base_y, base_z = 0, 0, 0
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            #base_x, base_y, base_z = landmark_point[0], landmark_point[1], landmark_point[2]
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        #temp_landmark_list[index][2] = (temp_landmark_list[index][2] - base_z)*200

    #print(temp_landmark_list) #21

    # 1次元リストに変換
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    # 正規化 ##########検討の余地あり
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n): return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def calc_distance_wrist2MFjoint(img, hand_landmarks):

    # 手首の座標を取得
    lm_0 = hand_landmarks.landmark[0]

    # 中指第一関節の座標を取得
    lm_9 = hand_landmarks.landmark[9]

    # 中指第一関節 - 手首の距離を計算
    distance_x = lm_0.x - lm_9.x
    distance_y = lm_0.y - lm_9.y
    distance_z = lm_0.z - lm_9.z # 角度の修正計算ができれば、ここは不要

    # 中指第一関節 - 手首の距離を代入
    distance_wrist2MFjoint = math.sqrt(distance_x ** 2 + distance_y ** 2 + distance_z ** 2)

    # - テキスト出力
    cv2.putText(
        img, 
        text=str(distance_wrist2MFjoint), 
        org=(200,50), 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=0.8, 
        color=(64, 0, 0), 
        thickness=1, 
        lineType=cv2.LINE_AA
    )

    return img, distance_wrist2MFjoint



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
    '''

    mainEstimate()