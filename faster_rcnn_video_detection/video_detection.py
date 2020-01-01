
import torchvision
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
import cv2
import time
import torch
import numpy as np
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
def get_prediction(img, threshold):

  img = Image.fromarray(np.uint8(img))
  transform = T.Compose([T.ToTensor()])
  img = transform(img)
  model.cuda()
  img = img.cuda()
  pred = model([img])
  print(pred)

  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list((pred[0]['labels']).to("cpu").numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list((pred[0]['boxes']).to("cpu").detach().numpy())]
  pred_score = list((pred[0]['scores']).to("cpu").detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return pred_boxes, pred_class

def object_detection_api(img, threshold=0.5, rect_th=12, text_size=15, text_th=3):
  boxes, pred_cls = get_prediction(img, threshold)
  # img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  for i in range(len(boxes)):
    cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
    cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX,  text_size, (0, 0, 128), thickness=text_th)
  return img
  # plt.figure(figsize=(20, 30))
  # plt.imshow(img)
  # plt.xticks([])
  # plt.yticks([])
  # plt.show()


if __name__ == '__main__':
    cam = cv2.VideoCapture("/home/dqq/下载/mz.mp4")# /home/dqq/下载/mz.mp4
    while cam.isOpened():
        start_time = time.time()
        ret_val, img = cam.read()
        composite = object_detection_api(img, rect_th=2, text_th=1, text_size=1)
        print("Time: {:.2f} s / img".format(time.time() - start_time))
        cv2.namedWindow("detections", 0)  # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
        cv2.resizeWindow("detections", 1000, 1000)  # 设置长和宽
        cv2.imshow("detections", composite,)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cam.release()
    cv2.destroyAllWindows()
