import torch
import torchvision
import argparse
import cv2
import sys

sys.path.append('./')
from faster_rcnn_video_detection import coco_names
import random

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Detection')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--dataset', default='coco', help='model')
    parser.add_argument('--score', type=float, default=0.5, help='objectness score threshold')
    args = parser.parse_args()

    return args


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    return (b, g, r)


def img_detection(src_img): #src_img from cv2.imread()
    with torch.no_grad():
        args = get_args()
        input = []
        num_classes = 91
        names = coco_names.names
        model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes, pretrained=True)
        model = model.cuda()
        model.eval()
        # src_img = cv2.imread(image_path)
        img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()
        input.append(img_tensor)
        out = model(input)
        boxes = out[0]['boxes']
        labels = out[0]['labels']
        scores = out[0]['scores']
        for idx in range(boxes.shape[0]):
            if scores[idx] >= args.score:
                x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
                name = names.get(str(labels[idx].item()))
                cv2.rectangle(src_img, (x1, y1), (x2, y2), random_color(), thickness=2)
                cv2.putText(src_img, text=name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, thickness=2, lineType=cv2.LINE_AA, color=(0, 0, 255))
        return src_img

def video_detection(video_path): #video_path = 0 means read cv2.VideoCapture(0)
    cam = cv2.VideoCapture(video_path)  # /home/dqq/下载/mz.mp4
    print("start...")
    while cam.isOpened():
        ret_val, img = cam.read()
        composite = img_detection(img)
        cv2.namedWindow("detections", 0)  # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
        cv2.resizeWindow("detections", 1906, 1080)  # 设置长和宽
        cv2.imshow("detections", composite, )
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_detection("/home/dqq/视频/test.mp4")