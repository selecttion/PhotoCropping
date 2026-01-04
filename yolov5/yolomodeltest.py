import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
import cv2  # 添加 OpenCV 库

def run_detection(source, weights):
    # 初始化设备（GPU 或 CPU）
    device = select_device('')  # 自动选择设备
    model = DetectMultiBackend(weights, device=device)  # 加载模型
    stride, names, pt = model.stride, model.names, model.pt

    # 加载数据
    dataset = LoadImages(source, img_size=640, stride=stride, auto=pt)
    for path, im, im0s, _, _ in dataset:
        im = torch.from_numpy(im).to(device)  # 转换为张量
        im = im.float() / 255.0  # 归一化
        if len(im.shape) == 3:
            im = im[None]  # 扩展维度

        # 推理
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)

        # 处理检测结果
        for i, det in enumerate(pred):
            p, im0 = Path(path), im0s.copy()
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # 缩放框
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

            # 保存结果
            save_path = str(Path('runs/detect') / p.name)
            im0 = annotator.result()
            cv2.imwrite(save_path, im0)  # 使用 OpenCV 保存图像
            print(f"结果已保存到: {save_path}")

if __name__ == '__main__':
    # 设置输入和权重路径
    source = 'data/images/bus.jpg'  # 输入图片路径
    weights = 'yolov5s.pt'  # 模型权重路径

    # 运行检测
    run_detection(source, weights)