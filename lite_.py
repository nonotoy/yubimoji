
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow import keras

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

dataset = './point_history.csv'
model_save_path = 'model/keypoint_classifier/gesture_classifier.hdf5'

# 指文字の数
NUM_CLASSES = 5

TIME_STEPS = 21
DIMENSION = 2

X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (TIME_STEPS * DIMENSION) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=42)
X_train = X_train.reshape(-1, TIME_STEPS, DIMENSION)
X_test = X_test.reshape(-1, TIME_STEPS, DIMENSION)


# 推論モデルを読み込み
model = tf.keras.models.load_model(model_save_path)
tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'

# モデルを変換(量子化)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# TensorFlow Liteでサポートされていない操作を含むモデルに対応するための設定
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False

tflite_quantized_model = converter.convert()
open(tflite_save_path, 'wb').write(tflite_quantized_model)

# 推論テスト
interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()

# 入出力テンソルを取得
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 推論実施の前に、入力データを適切な形状にリシェイプ
input_data = np.array([X_dataset[0]]).reshape(1, TIME_STEPS, DIMENSION)

# 推論実施
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])

#print(np.squeeze(tflite_results))
#print(np.argmax(np.squeeze(tflite_results)))