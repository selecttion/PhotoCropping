import cv2
import torch
import numpy as np
from pathlib import Path

# 加载 YOLOv5 模型
def load_model(weights_path):
    # 直接加载本地模型文件
    model = torch.hub.load('D:\work\PhotoCropping\yolov5-master', 'custom', path=weights_path, source='local')
    return model

# 绘制带颜色的检测框
def plot_boxes(image, results, class_id, color, conf_thres=0.4):
    for *xyxy, conf, cls in results.xyxy[0]:
        if int(cls) == class_id and conf >= conf_thres:  # 过滤类别和置信度
            label = f"{results.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
            cv2.putText(image, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# 主函数
def main(input_folder, output_folder):
    # 创建输出文件夹
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # 加载第一个模型（face 检测）
    face_model = load_model("runs/train/exp4/weights/best.pt")

    # 加载第二个模型（person 检测）
    person_model = load_model("yolov5s.pt")

    # 遍历输入文件夹中的所有图片
    input_folder = Path(input_folder)
    image_paths = list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.png"))  # 支持 jpg 和 png 格式

    for image_path in image_paths:
        # 加载图片
        image = cv2.imread(str(image_path))

        # 使用第一个模型检测 face
        face_results = face_model(image_path)
        print(f"看一下{face_results}")
        image = plot_boxes(image, face_results, class_id=0, color=(0, 0, 255), conf_thres=0.4)  # 红色框

        # 使用第二个模型检测 person
        person_results = person_model(image_path)
        image = plot_boxes(image, person_results, class_id=0, color=(255, 0, 0))  # 蓝色框（person 类别）

        # 保存结果
        # output_path = output_folder / image_path.name
        # cv2.imwrite(str(output_path), image)
        # print(f"结果已保存到 {output_path}")

if __name__ == "__main__":
    input_folder = "datasets/myphoto/origin"  # 替换为你的输入文件夹路径
    output_folder = "datasets/myphoto/my_results"  # 替换为你的输出文件夹路径
    main(input_folder, output_folder)