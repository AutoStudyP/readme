# sourcecode


import cv2
import sys
import time
import platform
import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

def cv2_draw_label(image, text, x, y):
    x, y = int(x), int(y)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = 'malgun.ttf'
    if platform.system() == 'Darwin': #맥
        font = 'AppleGothic.ttf'
    try:
        imageFont = ImageFont.truetype(font, 28)
    except:
        imageFont = ImageFont.load_default()
    draw.text((x, y), text, font=imageFont, fill=(255, 255, 255))
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return image

video_capture = cv2.VideoCapture(0)
time.sleep(3)
if not video_capture.isOpened():
    sys.exit()

##########모델 로드

interpreter = tf.lite.Interpreter(model_path='/home/pi/tflite1/Sample_TFLite_model/detect.tflite')

labels = []
f = open('/home/pi/tflite1/Sample_TFLite_model/labelmap.txt')
lines = f.readlines()
for line in lines:
    label = line.strip()
    labels.append(label)
f.close()
#print(len(labels)) #1001

##########모델 예측

while True:
    ret, image = video_capture.read()
    #print(type(image)) #<class 'numpy.ndarray'>
    #print(image.shape) #(720, 1280, 3)

    if not ret:
        break

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, dsize=(300, 300))
    numpy_image = np.array(rgb_image) #이미지 타입을 넘파이 타입으로 변환
    x_test = np.array([numpy_image])
    x_test = x_test.astype(dtype=np.uint8)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], x_test)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    y_predict = interpreter.get_tensor(output_details[0]['index'])
    #y_predict = (y_predict - 0) * 0.00390625
    y_predict = (y_predict - output_details[0]['quantization'][1]) * output_details[0]['quantization'][0]
    #y_predict = (y_predict - output_details[0]['quantization_parameters']['zero_points'][0]) * output_details[0]['quantization_parameters']['scales'][0]
    label = labels[y_predict[0].argmax()]
    confidence = y_predict[0][y_predict[0].argmax()]
    image = cv2_draw_label(image, '{} {:.2f}%'.format(label, confidence * 100), 30, 30)

    cv2.imshow('image', image)

    key = cv2.waitKey(1) #키 입력이 있을때 까지 1 밀리초 동안 대기
    if key == ord('q'): 
        break

video_capture.release()
cv2.destroyAllWindows()
if platform.system() == 'Darwin': #맥
    cv2.waitKey(1)
    
