
# Standard Library
import csv

# Third-Party Libraries
import numpy as np
import cv2

# Local Libraries
import draw
from draw import JpnText
import calc
from model import KeyPointClassifier

def classify(frame, landmarks_buffer, results_buffer):

    # 指文字のラベル
    labelFilePath = 'setting/hand_keypoint_classifier_label.csv'
    with open(labelFilePath, encoding='utf-8-sig') as f:
        Labels = csv.reader(f)
        Labels = [row[0] for row in Labels]

    # 初期化
    drawJpnText = JpnText(frame)

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

    # 文字の予測 ###############################################################
    # 予測モデルのロード
    tflitePath = "model/keypoint_classifier/keypoint_classifier.tflite"
    keypoint_classifier = KeyPointClassifier(tflitePath)

    # 予測
    lm_list = np.array(lm_normalised_buffer, dtype=np.float32)
    lm_list = np.expand_dims(lm_list, axis=0) # (1, 30, 40)

    yubimoji_id, confidence = keypoint_classifier(lm_list)

    # 画面表示 ################################################################
    # 判定結果と確信度を表示
    results_buffer.append([Labels[yubimoji_id], confidence])
    lastest_results = results_buffer[-30:] # 30フレーム分の履歴を保持 (画面に表示しきれなくなった場合は過去分から非表示)

    img_h = frame.shape[0]
    frame = drawJpnText.results(lastest_results, img_h)

    # ランドマーク間の線を表示
    lm_latest = landmarks_buffer[-1]
    frame = draw.lmLines(frame, lm_latest)

    # 手掌長の表示
    palmLength_latest = calc.palmLength(lm_latest)
    # frame = draw.palmLength(frame, palmLength_latest)

    confidence_threshold = 0.7

    if confidence > confidence_threshold:
        return frame, Labels[yubimoji_id], confidence
    
    else:
        yubimoji = 'N/A'
        low_confidence = 0
        return frame, yubimoji, low_confidence