import tensorflow as tf

# モデルのパス
tflite_model_path = "model/keypoint_classifier/keypoint_classifier.tflite"

# TensorFlow Lite インタープリタの初期化
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# 入力ディテールの取得
input_details = interpreter.get_input_details()
print(input_details[0]['shape'])  # 入力テンソルの形状を表示