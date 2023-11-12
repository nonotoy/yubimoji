import pandas as pd
import glob

# パスで指定したファイルの一覧をリスト形式で取得. （ここでは一階層下のtestファイル以下）
csv_files = glob.glob('point_history/*.csv')

#読み込むファイルのリストを表示
for a in csv_files:
    print(a)

#csvファイルの中身を追加していくリストを用意
data_list = []

#読み込むファイルのリストを走査
for file in csv_files:
    print(file)
    data_list.append(pd.read_csv(file, header=None))

#リストを全て行方向に結合
df = pd.concat(data_list, axis=0, ignore_index=True)

df.to_csv("./point_history.csv",index=False, header=False)