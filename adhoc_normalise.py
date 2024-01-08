##############################################################################
# 正規化していないlandmarksを後から正規化する際に使う
##############################################################################

# Standard Library
import csv
import glob

# Third-Party Libraries
import pandas as pd
import numpy as np

# Local Libraries
import calc
import write


def combine_csv(saveto):

    # パスで指定したファイルの一覧をリスト形式で取得. （ここでは一階層下のtestファイル以下）
    csv_files = glob.glob('./point_history/*.csv')

    buffer_size = 30 # バッファリングするフレーム数

    #csvファイルの中身を追加していくリストを用意
    data_list = []

    #読み込むファイルのリストを走査
    for file in csv_files:

        # ファイルを読み込む
        csv_ = pd.read_csv(file, header=None)
        
        if len(csv_) < buffer_size:
            continue

        elif len(csv_) > buffer_size:
            csv_ = csv_.tail(buffer_size)
            data_list.append(csv_)

    #リストを全て行方向に結合
    df = pd.concat(data_list, axis=0, ignore_index=True)

    df.to_csv(saveto,index=False, header=False)


# calc.pyの同名関数と同じだが、入力するランドマークがリスト形式
def lmRelativeLoc(img_width, img_height, landmarks):

    # 画面上のランドマークの位置を算出
    for id, landmark in enumerate(landmarks):
        # x軸
        if np.mod(id,2) == 0:
            landmarks[id] = min(int(landmark * img_width), img_width - 1)
        # y軸
        elif np.mod(id,2) == 1:
            landmarks[id] = min(int(landmark * img_height), img_height - 1)

    landmark_list = [[landmarks[i], landmarks[i+1]] for i in range(0, len(landmarks), 2)]

    return landmark_list



def adhocNormalise(csv_path):

    with open(csv_path, encoding='utf-8-sig') as f:

        # 結合後のcsvを取得
        landmarks_list = csv.reader(f)

        for landmarks in landmarks_list:

            landmark_tmp = []

            # 一列目から指文字IDを取得、それ以降はランドマークを取得し、リストに格納
            for i in range(len(landmarks)):
                if i == 0:
                    yubimoji = landmarks[i]
                else:
                    landmark_tmp.append(float(landmarks[i]))

            # 正規化 ##################################################################
            landmark_list = lmRelativeLoc(640, 480, landmark_tmp)

            # 手掌長で正規化する場合
            #lm_normalised = calc.Normalisation(landmark_list, palmLength) 

            lm_normalised = calc.Normalisation(landmark_list, ignore_normalise=False)
            
            # 書き出し ################################################################
            write.csvRecord(lm_normalised,
                            yubimoji,
                            csv_path='./point_history_normalised.csv')



csv_path = './point_history.csv'

# 結合前のcsvを取得
combine_csv(saveto=csv_path)

adhocNormalise(csv_path)

print('Done')