import os
import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5x.pt', device='cpu')
model.cpu()

model.conf = 0.10
model.iou = 0.10
model.agnostic = False
model.multi_label = False
model.classes = [0]  # FILTRANDO POR CLASSE , NO NOSSO CASO 0
model.max_det = 1000
model.amp = False

path_images_to_process = './img'
path_objects_detected = "./detected"

for filename in os.listdir(path_images_to_process):
    f = os.path.join(path_images_to_process, filename)
    image = cv2.imread(f)
    results = model(image)

    if len(results.xyxy[0]) > 1:
        print(f"TEM MAIS DE UMA PESSOA NA IMAGEM: {f}")

    if len(results.xyxy[0]) > 0 and results.xyxy[0][0][5] == 0:
        print(f"UMA PESSOA FOI DETECTADA NA IMAGEM: {f}")


