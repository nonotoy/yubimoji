# ランドマークの座標をcsvに保存する

import csv
import datetime

# 各ランドマークの座標を取得
def reshapeLandmark(landmarks):

    # landmark_list = [lm.x for lm in landmarks.landmark] + [lm.y for lm in landmarks.landmark]
    landmark_list = []

    for i in range(len(landmarks.landmark)):
        lm = landmarks.landmark[i]
        landmark_list.append(lm.x)
        landmark_list.append(lm.y)

    return landmark_list


# csv保存
def csvRecord(landmark_list, yubimoji_id=None, starttime=None):

    starttime = starttime.strftime('%Y%m%d%H%M%S') if starttime != None else datetime.datetime.now().strftime('%Y%m%d%H%M%S') #現在時刻の取得

    if yubimoji_id == None:
        csv_path = './point_history_{0}.csv'.format(starttime)
    else:
        csv_path = './point_history_{0}_{1}.csv'.format(str(yubimoji_id).zfill(2),starttime)

    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([yubimoji_id, *landmark_list])