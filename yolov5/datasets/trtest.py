import os
import random

import cv2


def draw_boxes(image_path, label_path):
    """
    在图像上绘制边界框并显示。

    参数:
        image_path: 图像文件路径
        label_path: 标注文件路径
    """
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法加载图像 {image_path}")
        return

    # 获取图像尺寸
    img_height, img_width, _ = image.shape

    # 加载标注文件
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # 绘制边界框
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())

        # 将YOLOv5格式转换为边界框坐标
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)

        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框，线宽为2

    # 显示图像
    cv2.imshow('Image with Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def verify_labels(image_dir, label_dir, num_samples=5):
    """
    随机选择几张图像和对应的标注文件，验证标注是否正确。

    参数:
        image_dir: 图像目录
        label_dir: 标注目录
        num_samples: 需要验证的图像数量
    """
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    if not image_files:
        print(f"错误: 目录 {image_dir} 中没有图像文件")
        return

    # 随机选择几张图像
    selected_images = random.sample(image_files, min(num_samples, len(image_files)))

    # 验证每张图像
    for image_name in selected_images:
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt'))

        if not os.path.exists(label_path):
            print(f"警告: 标注文件不存在 {label_path}")
            continue

        print(f"正在验证图像: {image_name}")
        draw_boxes(image_path, label_path)


# 示例：验证训练集
verify_labels(
    image_dir='widerface/images/train',
    label_dir='widerface/labels/train',
    num_samples=5
)