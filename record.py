
# Standard Library
import csv

# Third-Party Libraries
import cv2

# Local Libraries
import draw
import calc
import write

def append(frame, landmarks, yubimoji_id, recordsCnt, starttime):

    # 録画位置指示
    recLocations = ['正面','左上','上','右上','左','正面','右','左下','下','右下']

    # 指文字のラベル
    labelFilePath = 'setting/hand_keypoint_classifier_label.csv'
    with open(labelFilePath, encoding='utf-8-sig') as f:
        Labels = csv.reader(f)
        Labels = [row[0] for row in Labels]

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

    return frame


# 各ランドマークの座標を取得
def reshapeLandmark(landmarks):

    landmark_list = []

    for i in range(len(landmarks.landmark)):
        lm = landmarks.landmark[i]
        landmark_list.append(lm.x)
        landmark_list.append(lm.y)

    return landmark_list
