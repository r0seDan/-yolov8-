#coding:utf-8
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolov8n.pt")
# Use the model
if __name__ == '__main__':
    # Use the model
    results = model.train(data=r'D:\py\py_pro\YOLOV8System\CarPersonDetection\datasets\CarPersonData\data.yaml', epochs=250, batch=4)  # 训练模型




