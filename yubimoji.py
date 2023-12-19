# Outputとなるクラス数: 87
# 清音46 + 濁音・半濁音・拗音・長音・促音30 + 数字10 + コマンド1 (入力削除用) 

import math
import csv
import os
import sys
import cv2
import copy
import datetime
import itertools
import time
import collections
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from model import KeyPointClassifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

recordsCnt_list = [' 正面',' 左上',' 上',' 右上',' 左',' 正面',' 右',' 左下',' 下',' 右下'] # 録画位置指示

# 指文字のラベル
with open('setting/hand_keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    yubimoji_labels = csv.reader(f)
    yubimoji_labels = [row[0] for row in yubimoji_labels]


def main(mode, yubimoji_id=None):
    # mode: 0 -> 録画モード, 1 -> 予測モード

    if mode == 0:
        # (録画モードのみ) 録画用変数
        recordsCnt = 0 # カウント

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

    # カメラキャプチャ設定
    video_capture = cv2.VideoCapture(0) #内臓カメラ
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    # バッファリング用
    #fps = video_capture.get(cv2.CAP_PROP_FPS) # フレームレートの取得
    #buffer_duration = 3  # バッファリングする秒数
    #buffer_size = int(fps * buffer_duration) # バッファリングするフレーム数
    buffer_size = 30 # バッファリングするフレーム数
    buffering_threshold = buffer_size // 2 # バッファリングを開始する閾値

    # ランドマークのバッファ（リングバッファ）の初期化
    landmarks_buffer = collections.deque(maxlen=buffer_size)

    # 手掌長のバッファ
    palmsize_buffer = collections.deque(maxlen=buffer_size)

    startprocessbufferno = None

    starttime = datetime.datetime.now() #現在時刻の取得

    if video_capture.isOpened():

        while True:

            processing = False

            # (録画モードのみ) 録画回数カウント
            if mode == 0:
                sec_dif = datetime.datetime.now() - starttime
                sec_dif = sec_dif.total_seconds()  
                if sec_dif > 3:
                    time.sleep(1)
                    starttime = datetime.datetime.now() 
                    recordsCnt += 1

                if recordsCnt >= len(recordsCnt_list):
                    break

            # キー入力(ESC:プログラム終了)
            key = cv2.waitKey(1)
            if key == 27: break

            # カメラから画像取得
            ret, frame = video_capture.read()
            if not ret: break
    
            # 画像を左右反転
            frame = cv2.flip(frame, 1)

            # バッファ内のランドマークをLSTMモデルに供給
            # 検出処理の実行
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:

                for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                    # 手の左右判定 / 今回の副テーマ研究では左手の対応はしない
                    if handedness.classification[0].label[0:] == 'Right':

                        # 録画モード // バッファは貯めずにタイムフレーム毎に処理
                        if mode == 0: 

                            appendRecord(frame, landmarks, yubimoji_id, recordsCnt, starttime)

                        # 判定モード // 規定バッファ数貯めてから処理
                        elif mode == 1:

                            # 入力開始動作の検出 // 手を閾値以上に前面に動かした時に入力開始の合図とする
                            # 15フレーム分を一度別にバッファリングしておいて、その中で平均的に前に動いている場合
                            # もしくは最初と最後を比較して閾値より前面に動いている場合はバッファリングをためて、判定器へ流す

                            # 手掌長を計算してから専用バッファを加えるのでいいかも。
                            palmsize = calc_palmsize(landmarks)
                            palmsize_buffer.append(palmsize)
                            
                            #if len(palmsize_buffer) < buffering_threshold:
                            landmarks_buffer.append(landmarks)

                            if detectInputGesture(palmsize_buffer):

                                # 入力開始動作を検知したフレーム番号を取得・更新
                                startprocessbufferno = len(landmarks_buffer) - buffering_threshold + 1

                                if len(landmarks_buffer) >= buffering_threshold:

                                    processBuffer(
                                        frame, 
                                        landmarks_buffer
                                    )
                                    print('処理A',startprocessbufferno, len(landmarks_buffer))
                                
                            elif startprocessbufferno != None:      
                                
                                processBuffer(
                                    frame, 
                                    landmarks_buffer
                                )
                                
                                print('処理B',startprocessbufferno, len(landmarks_buffer))

                                # 入力開始動作を検知したフレーム番号がない場合は無視
                                # 入力開始動作をしたけど、新しいフレームが一定数入ってこない場合はバッファをクリアさせたい。例えば60フレーム

    # リソースの解放
    video_capture.release()
    hands.close()
    cv2.destroyAllWindows()



def appendRecord(frame, landmarks, yubimoji_id, recordsCnt, starttime):

    # 手掌長の取得  (Priyaら (2023)) ######################################################
    # 手掌長 (中指第一関節 - 手首) 計算
    palmsize = calc_palmsize(landmarks)

    # 画面表示: 手掌長の表示
    # cv2.putText(landmarks_buffer, text=str(palmsize), org=(200,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(64, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    # 正規化 ##############################################################################
    # ランドマークの画像上の位置 (x,y座標) を算出する関数 (z座標が必要になる場合は関数内で調整)
    # 理由: 次のブロックで実施する相対座標の計算のため
    landmark_list = calc_landmark_list(frame, landmarks)

    # 手掌長で正規化
    # 理由: 学習データに無い位置でジェスチャーをした場合、認識が上手くいかないため、一番最初の座標をもとに相対座標を取得する
    #lm_normalised = pre_process_landmark(landmark_list, palmsize)
    lm_normalised = pre_process_landmark(landmark_list)

    # 画面表示 ###########################################################################
    # 文字表示
    str_recPosition = recordsCnt_list[recordsCnt] if yubimoji_id != None else ''
    frame = putText_japanese(frame, yubimoji_labels[yubimoji_id] + str_recPosition)
    print(yubimoji_labels[yubimoji_id])

    # ランドマーク間の線を表示
    frame = showLandmarkLines(frame, landmarks)

    # 録画のみ: 検出情報をcsv出力 ##########################################################
    lm_reshaped = reshapeLandmark(landmarks)
    write_csv(yubimoji_id, lm_reshaped, starttime)
    write_csv(yubimoji_id, lm_normalised, starttime)  

    # 画像の表示
    cv2.imshow("MediaPipe Hands", frame)



def processBuffer(frame, landmarks_buffer):

    lm_normalised_buffer = []

    # 予測モデルのロード
    keypoint_classifier = KeyPointClassifier("model/keypoint_classifier/keypoint_classifier.tflite")

    # バッファ内の各フレームに対する処理
    for landmarks in landmarks_buffer:

        # 手掌長の取得  (Priyaら (2023)) ######################################################
        # 手掌長 (中指第一関節 - 手首) 計算
        palmsize = calc_palmsize(landmarks)

        # 画面表示: 手掌長の表示
        # cv2.putText(landmarks_buffer, text=str(palmsize), org=(200,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(64, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        # 正規化 ##############################################################################
        # ランドマークの画像上の位置 (x,y座標) を算出する関数 (z座標が必要になる場合は関数内で調整)
        # 理由: 次のブロックで実施する相対座標の計算のため
        landmark_list = calc_landmark_list(frame, landmarks)
        
        # 手掌長で正規化
        # 理由: 学習データに無い位置でジェスチャーをした場合、認識が上手くいかないため、一番最初の座標をもとに相対座標を取得する
        #lm_normalised = pre_process_landmark(landmark_list, palmsize)
        lm_normalised = pre_process_landmark(landmark_list)

        # 正規化後のランドマークをバッファごとに保管
        lm_normalised_buffer.append(lm_normalised)

    # 文字ラベルの予測 #####################################################################
    # lm_normalised_bufferのサイズを確認    
    print(len(lm_normalised_buffer))

'''
    # 予測器に合うように形状を変更
    lm_normalised = np.array(lm_normalised_buffer).reshape((1, 30, 40))
    # 予測
    yubimoji_id = keypoint_classifier(lm_normalised)

    # 画面表示 ###########################################################################
    # 文字表示
    frame = putText_japanese(landmarks_buffer, yubimoji_labels[yubimoji_id])
    print(yubimoji_labels[yubimoji_id])

    # ランドマーク間の線を表示
    frame = showLandmarkLines(frame, landmarks)

    # 画像の表示
    cv2.imshow("MediaPipe Hands", frame)
'''



def calc_landmark_list(frame, landmarks):

    img_width, img_height = frame.shape[1], frame.shape[0]

    landmark_list = []

    # 画面上のランドマークの位置を算出
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * img_width), img_width - 1)
        landmark_y = min(int(landmark.y * img_height), img_height - 1)
        landmark_list.append([landmark_x, landmark_y])

    return landmark_list


def pre_process_landmark(landmark_list, palmsize=None):

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

    # 最初の二つは0なので削除
    del temp_landmark_list[0:2]

    if palmsize != None:
        # 手掌長で正規化
        def normalize_(n): return n / palmsize
        temp_landmark_list = list(map(normalize_, temp_landmark_list))

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
def putText_japanese(frame, text):

    #Notoフォント
    font = ImageFont.truetype('/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc', size=20)

    #imgをndarrayからPILに変換
    img_pil = Image.fromarray(frame)

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
def showLandmarkLines(frame, landmarks):

    # 各ランドマークの座標はlandmarks.landmark[0]~[20].x, .y, .zに格納されている
    img_h, img_w, _ = frame.shape

    # Landmark間の線
    landmark_line_ids = [ 
        (0, 1),     # 手首 - 親指第一関節
        (1, 5),     # 親指第一関節 - 人差し指第一関節
        (5, 9),     # 人差し指第一関節 - 中指第一関節
        (9, 13),    # 中指第一関節 - 薬指第一関節
        (13, 17),   # 薬指第一関節 - 小指第一関節
        (17, 0),    # 小指第一関節 - 手首
        (0, 9),     # 中指第一関節 - 手首
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
        cv2.line(frame, lm_pos1, lm_pos2, (128, 0, 0), 1)

    # landmarkをcircleで表示
    z_list = [lm.z for lm in landmarks.landmark]
    z_min = min(z_list)
    z_max = max(z_list)
    for lm in landmarks.landmark:
        lm_pos = (int(lm.x * img_w), int(lm.y * img_h))
        lm_z = int((lm.z - z_min) / (z_max - z_min) * 255)
        cv2.circle(
            frame,
            radius=3, 
            center=lm_pos, 
            color=(255, lm_z, lm_z), 
            thickness=-1
        )

    return frame

# 各ランドマークの座標を取得
def reshapeLandmark(landmarks):

    # landmark_list = [lm.x for lm in landmarks.landmark] + [lm.y for lm in landmarks.landmark]
    landmark_list = []

    for i in range(len(landmarks.landmark)):
        lm = landmarks.landmark[i]
        landmark_list.append(lm.x)
        landmark_list.append(lm.y)

    print(landmark_list)

    return landmark_list


# csv保存
def write_csv(yubimoji_id, landmark_list, starttime):

    starttime = starttime.strftime('%Y%m%d%H%M%S')

    csv_path = './point_history_{0}_{1}.csv'.format(str(yubimoji_id).zfill(2),starttime)

    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([yubimoji_id, *landmark_list])


# 入力動作の検出
def detectInputGesture(palmsize_buffer, threshold=0.1):

    if len(palmsize_buffer) < 15:
        return False

    else:
        cur_palmlength = palmsize_buffer[14]
        prev_palmlength = palmsize_buffer[0]

        # ランドマーク間の距離を計算
        distances = cur_palmlength - prev_palmlength

        # ランドマーク間の距離の最大値が閾値を超えたらTrueを返す
        return distances > threshold



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
    char = 70
    # 50cm

    #main(mode,char) # Record
    main(mode) # Predict