##############################################################################
# Description: 日本手話の指文字を認識するプログラム
##############################################################################

# Standard Library
import os
import sys
import datetime
import time
import collections

# Third-Party Libraries
import cv2
import mediapipe as mp

# Local Libraries
import calc
import record
import classify

# 共通変数 ####################################################################
frameCount = 0

InputCount = 0 # 検査用
noInputCount = 0 # 検査用

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
    fps = video_capture.get(cv2.CAP_PROP_FPS) # フレームレートの取得
    buffer_duration = 3  # バッファリングする秒数
    buffer_size = 30 # int(fps * buffer_duration) # バッファリングするフレーム数

    # フレームカウント
    frameCount = 0
    InputCount = 0 # 検査用

    # ランドマークのバッファ（リングバッファ）の初期化
    landmarks_buffer = collections.deque(maxlen=buffer_size)

    # 確認用: 処理中のバッファ番号保管用メタバッファ
    processingNo_buffer = collections.deque(maxlen=buffer_size)

    # 手掌長のバッファ
    palmLength_buffer = collections.deque(maxlen=buffer_size)

    # 結果のバッファ
    results_buffer = []

    # 処理中フラグ
    isProcessing = False

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

                if recordsCnt >= 9: #len(recLocations)
                    break

            # キー入力(ESC:プログラム終了)
            key = cv2.waitKey(1)
            if key == 27: break

            # カメラから画像取得
            ret, frame = video_capture.read()
            if not ret: break
    
            # 画像を左右反転
            frame = cv2.flip(frame, 1)

            # 判定モードの場合、結果表示領域用にframe内の指定範囲を塗りつぶし
            if mode == 1:
                height, width = frame.shape[:2]
                colour = (255, 255, 255) # 白
                cv2.rectangle(frame, (550, 0), (width, height), colour, -1)

            # FPSの表示
            cv2.putText(frame, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

            # 検出処理の実行
            ##################################################################
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:

                for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                    # 手の左右判定 / 今回の副テーマ研究では左手の対応はしない
                    if handedness.classification[0].label[0:] == 'Right':

                        # 非入力カウントのリセット
                        noInputCount = 0

                        # チェック用
                        InputCount += 1

                        # 録画モード // バッファは貯めずにタイムフレーム毎に処理
                        if mode == 0: 

                            frame = record.append(
                                frame, 
                                landmarks, 
                                yubimoji_id, 
                                recordsCnt, 
                                starttime)

                        # 判定モード // 規定バッファ数貯めてから処理
                        elif mode == 1:

                            # 入力開始動作の検出 // 手を閾値以上に前面に動かした時に入力開始の合図とする
                            # 15フレーム分を一度別にバッファリングしておいて、その中で平均的に前に動いている場合
                            # もしくは最初と最後を比較して閾値より前面に動いている場合はバッファリングをためて、判定器へ流す
                            


                            
                            # メインバッファを作って、そこから各処理用にバッファを切り分ける



                            # 手掌長を計算してから専用バッファに追加
                            palmLength = calc.palmLength(landmarks)
                            palmLength_buffer.append(palmLength)
                            
                            landmarks_buffer.append(landmarks) # 最新の30フレーム分をバッファリング / 古いものは自動消去される

                            # 確認用
                            processingNo_buffer.append(frameCount) # バッファリングしたフレーム番号を保存 / 古いものは自動消去される
                            # 正しく過去30フレーム分が判定処理に流れていることを確認

                            # バッファが必要数貯まったら処理
                            if len(landmarks_buffer) >= buffer_size:

                                # 入力開始動作を検知した場合
                                if detectInputGesture(palmLength_buffer):

                                    # 入力処理フラグを立てる
                                    isProcessing = True

                                    # 入力フラグを立てた時点でのカウントを保存
                                    processingBufferCnt = InputCount

                                    frame, yubimoji, confidence = classify.classify(frame,
                                                                                    landmarks_buffer,
                                                                                    results_buffer)
                                    
                                    print(frameCount, yubimoji, '{:.2f}%'.format(confidence * 100))

                                # すでに処理中の場合
                                elif isProcessing == True:
                                    
                                    # 判定処理中でも閾値以上入力開始動作の検知がなく、判定処理が規定フレーム以上続いている場合、変化なしと見做してバッファをクリア
                                    # 意図通り動かないので、一度無効化
                                    '''
                                    clearance_threshold = 5000000000
                                    if InputCount - processingBufferCnt > clearance_threshold:

                                        clearBuffer(landmarks_buffer, 
                                                    palmLength_buffer, 
                                                    results_buffer, 
                                                    clearance_threshold=clearance_threshold)
                                        
                                    else:
                                    '''

                                    frame, yubimoji, confidence = classify.classify(frame,
                                                                                    landmarks_buffer,
                                                                                    results_buffer)
                                    
                                    print(frameCount, yubimoji, '{:.2f}%'.format(confidence * 100))

                            # バッファが溜まっていない場合は処理なし
                            else:
                                print(frameCount, ':Buffering')
                                    
                    # 左手の場合
                    else:
                        clearBuffer(landmarks_buffer, 
                                    processingNo_buffer,
                                    palmLength_buffer, 
                                    results_buffer)

            # 入力がない場合
            else:
                clearBuffer(landmarks_buffer, 
                            processingNo_buffer,
                            palmLength_buffer, 
                            results_buffer)
                
            # 画像の表示
            cv2.imshow("MediaPipe Hands", frame)
            # print(processingNo_buffer)
            frameCount += 1


    # リソースの解放
    video_capture.release()
    hands.close()
    cv2.destroyAllWindows()


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


# バッファのクリア
def clearBuffer(landmarks_buffer, processingNo_buffer, palmLength_buffer, results_buffer, clearance_threshold=15):

    global isProcessing, frameCount, noInputCount, InputCount

    noInputCount += 1

    if noInputCount > clearance_threshold:
        # 新しいフレームが一定数入ってこない場合はバッファをクリア
        # 判定の結果右手の判定が数フレーム外れてしまうことも考えられるので、閾値を超えるまではクリアしない
        landmarks_buffer.clear()
        processingNo_buffer.clear()
        palmLength_buffer.clear()
        results_buffer.clear()

        isProcessing = False
        noInputCount = 0
        InputCount = 0

        print('{0}: Buffer & booleans cleared'.format(frameCount))


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