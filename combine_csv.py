import pandas as pd
import glob

# パスで指定したファイルの一覧をリスト形式で取得. （ここでは一階層下のtestファイル以下）
csv_files = glob.glob('point_history/*.csv')

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
        print(len(csv_))

#リストを全て行方向に結合
df = pd.concat(data_list, axis=0, ignore_index=True)
print(len(df))

df.to_csv("./point_history.csv",index=False, header=False)