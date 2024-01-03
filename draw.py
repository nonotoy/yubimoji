# opencvのフレームに対して、各種描画を行う関数を定義

import cv2

import numpy as np
from PIL import ImageFont, ImageDraw, Image

# 日本語表示
def jpntext(frame, text):

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
def lmLines(frame, landmarks):

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


# 手掌長の表示
def palmLength(frame, palmLength):

    cv2.putText(
        frame,
        text=str(palmLength), 
        org=(200,50), 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=0.8, 
        color=(64, 0, 0), 
        thickness=1, 
        lineType=cv2.LINE_AA
    )

    return frame