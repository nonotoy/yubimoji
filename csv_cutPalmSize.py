import os
import pandas as pd

# 処理するフォルダのパス
folder_path = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/yubimoji/point_history'

# フォルダ内のすべてのCSVファイルを取得
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)

        # CSVファイルを読み込む
        df = pd.read_csv(file_path)

        # 最終列を除外する
        df = df.iloc[:, :-1]

        # 変更をCSVファイルに書き戻す
        df.to_csv(file_path, index=False)