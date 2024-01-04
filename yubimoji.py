##############################################################################
# Description: 日本手話の指文字を認識するプログラム
##############################################################################

# Standard Library
import os
import sys
import csv
import datetime
import time
import collections

# Third-Party Libraries
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf

# Local Libraries
import draw
import calc
import write
from model import KeyPointClassifier

# 共通変数 ####################################################################

# 録画位置指示
recLocations = ['正面','左上','上','右上','左','正面','右','左下','下','右下']

# 指文字のラベル
labelFilePath = 'setting/hand_keypoint_classifier_label.csv'
with open(labelFilePath, encoding='utf-8-sig') as f:
    Labels = csv.reader(f)
    Labels = [row[0] for row in Labels]


# メイン関数 ##################################################################
def main(mode, yubimoji_id=None):
    # mode: 0 -> 録画モード, 1 -> 予測モード

    # (録画モードのみ) 録画用カウント変数の設定
    if mode == 0:
        if yubimoji_id != None and yubimoji_id >= 0 and yubimoji_id <= 88:
            recordsCnt = 0

        # yubimoji_idが存在しない、もしくは空の場合、停止
        else:
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
    buffer_size = 30 # int(fps * buffer_duration) バッファリングするフレーム数
    buffering_threshold = buffer_size // 2 # バッファリングを開始する閾値

    # ランドマークのバッファ（リングバッファ）の初期化
    landmarks_buffer = collections.deque(maxlen=buffer_size)

    # 手掌長のバッファ
    palmLength_buffer = collections.deque(maxlen=buffer_size)

    startprocessbufferno = None

    starttime = datetime.datetime.now() #現在時刻の取得

    if video_capture.isOpened():

        while True:

            # (録画モードのみ) 録画回数カウント
            if mode == 0:
                sec_dif = datetime.datetime.now() - starttime
                sec_dif = sec_dif.total_seconds()  
                if sec_dif > 3:
                    time.sleep(1)
                    starttime = datetime.datetime.now() 
                    recordsCnt += 1

                if recordsCnt >= len(recLocations):
                    break

            # キー入力(ESC:プログラム終了)
            key = cv2.waitKey(1)
            if key == 27: break

            # カメラから画像取得
            ret, frame = video_capture.read()
            if not ret: break
    
            # 画像を左右反転
            frame = cv2.flip(frame, 1)

            # 以下を関数化 frame, landmarks_buffer, starttimeを引数に

            # if mode == 0: 
            # performRecognition(frame, starttime, yubimoji_id, recordsCnt)
            # else:
            # performRecognition(frame, landmarks_buffer, palmLength_buffer, starttime)


            # 検出処理の実行
            ##################################################################
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:

                for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                    # 手の左右判定 / 今回の副テーマ研究では左手の対応はしない
                    if handedness.classification[0].label[0:] == 'Right':

                        # 録画モード // バッファは貯めずにタイムフレーム毎に処理
                        if mode == 0: 

                            appendRecord(frame, 
                                         landmarks, 
                                         yubimoji_id, 
                                         recordsCnt, 
                                         starttime)

                        # 判定モード // 規定バッファ数貯めてから処理
                        elif mode == 1:

                            # 入力開始動作の検出 // 手を閾値以上に前面に動かした時に入力開始の合図とする
                            # 15フレーム分を一度別にバッファリングしておいて、その中で平均的に前に動いている場合
                            # もしくは最初と最後を比較して閾値より前面に動いている場合はバッファリングをためて、判定器へ流す

                            # 手掌長を計算してから専用バッファを加えるのでいいかも。
                            palmLength = calc.palmLength(landmarks)
                            palmLength_buffer.append(palmLength)
                            
                            #if len(palmLength_buffer) < buffering_threshold:
                            landmarks_buffer.append(landmarks)

                            if (detectInputGesture(palmLength_buffer) and len(landmarks_buffer) >= 30) or startprocessbufferno != None:

                                # 入力開始動作を検知したフレーム番号を取得・更新
                                if startprocessbufferno == None:
                                    startprocessbufferno = len(landmarks_buffer) - buffering_threshold + 1

                                processBuffer(frame, landmarks_buffer, starttime)
                                #print(len(landmarks_buffer))

                                #print(startprocessbufferno, len(landmarks_buffer))

                                # 入力開始動作を検知したフレーム番号がない場合は無視
                                # 入力開始動作をしたけど、新しいフレームが一定数入ってこない場合はバッファをクリアさせたい。例えば60フレーム

    # リソースの解放
    video_capture.release()
    hands.close()
    cv2.destroyAllWindows()


def appendRecord(frame, landmarks, yubimoji_id, recordsCnt, starttime):

    # 手掌長の取得 #############################################################
    palmLength = calc.palmLength(landmarks)

    # 正規化 ##################################################################
    landmark_list = calc.lmRelativeLoc(frame, landmarks)

    # 手掌長で正規化する場合
    #lm_normalised = calc.Normalisation(landmark_list, palmLength) 

    lm_normalised = calc.Normalisation(landmark_list)

    # 画面表示 ################################################################

    # 文字表示
    str_recPosition = recLocations[recordsCnt] if yubimoji_id != None else ''
    frame = draw.jpntext(frame, Labels[yubimoji_id] + str_recPosition)

    # ランドマーク間の線を表示
    frame = draw.lmLines(frame, landmarks)

    # 手掌長の表示
    # frame = draw.palmLength(frame, palmLength)

    # 録画のみ: 検出情報をcsv出力 ###############################################

    lm_reshaped = reshapeLandmark(landmarks)
    write.csvRecord(lm_reshaped, yubimoji_id, starttime)
    write.csvRecord(lm_normalised, yubimoji_id, starttime)  

    # 画像の表示
    cv2.imshow("MediaPipe Hands", frame)


def processBuffer(frame, landmarks_buffer, starttime=None):

    lm_normalised_buffer = []

    # バッファ内の各フレームに対する処理
    for landmarks in landmarks_buffer:

        # 手掌長の取得 #########################################################
        palmLength = calc.palmLength(landmarks)

        # 正規化 ##############################################################
        landmark_list = calc.lmRelativeLoc(frame, landmarks)
        # 手掌長で正規化する場合
        #lm_normalised = calc.Normalisation(landmark_list, palmLength) 
        lm_normalised = calc.Normalisation(landmark_list)

        # 正規化後のランドマークをバッファごとに保管
        lm_normalised_buffer.append(lm_normalised)

        # 画面表示 ############################################################
        # ランドマーク間の線を表示
        frame = draw.lmLines(frame, landmarks)

        # 手掌長の表示
        # frame = draw.palmLength(frame, palmLength)

    # 文字の予測 ###############################################################
    # 予測モデルのロード
    tflitePath = "model/keypoint_classifier/keypoint_classifier.tflite"
    keypoint_classifier = KeyPointClassifier(tflitePath)

    # 予測
    lm_list = np.array(lm_normalised_buffer, dtype=np.float32)
    lm_list = np.expand_dims(lm_list, axis=0)

    yubimoji_id = keypoint_classifier(lm_list)

    # 画面表示 ################################################################
    # 文字表示
    frame = draw.jpntext(frame, Labels[yubimoji_id])

    # データが流れているかの確認用
    #lm_reshaped = reshapeLandmark(landmarks_buffer)
    #write.csvRecord(landmarks_buffer) #30フレーム

    # 画像の表示
    cv2.imshow("MediaPipe Hands", frame)


# 各ランドマークの座標を取得
def reshapeLandmark(landmarks):

    landmark_list = []

    for i in range(len(landmarks.landmark)):
        lm = landmarks.landmark[i]
        landmark_list.append(lm.x)
        landmark_list.append(lm.y)

    return landmark_list


# 入力動作の検出
def detectInputGesture(palmLength_buffer, threshold=0.1):

    if len(palmLength_buffer) < 15:
        return False

    else:
        cur_palmLength = palmLength_buffer[14]
        prev_palmLength = palmLength_buffer[0]

        # ランドマーク間の距離を計算
        distances = cur_palmLength - prev_palmLength

        # ランドマーク間の距離の最大値が閾値を超えたらTrueを返す
        return distances > threshold


if __name__ == "__main__": 
    # Outputとなるクラス数: 87
    # 清音46 + 濁音・半濁音・拗音・長音・促音30 + 数字10 + コマンド1 (入力削除用) 

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

    # mode: 0 -> 録画モード, 1 -> 予測モード
    mode = 1
    char = 70
    # 50cm

    #main(mode,char) # Record
    main(mode) # Predict