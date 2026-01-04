import os
import random
import shutil
import cv2

def sample_images_and_convert_labels(src_image_dir, src_label_file, dest_image_dir, dest_label_dir, sample_ratio=0.1):
    """
    从源目录中随机抽样图像，并转换对应的标注文件。

    参数:
        src_image_dir: 源图像目录（包含子文件夹）
        src_label_file: 源标注文件路径
        dest_image_dir: 目标图像目录
        dest_label_dir: 目标标注目录
        sample_ratio: 抽样比例（默认10%）
    """
    # 检查源标注文件是否存在
    if not os.path.exists(src_label_file):
        raise FileNotFoundError(f"标注文件不存在: {src_label_file}")

    # 检查源图像目录是否存在
    if not os.path.exists(src_image_dir):
        raise FileNotFoundError(f"图像目录不存在: {src_image_dir}")

    # 创建目标目录
    os.makedirs(dest_image_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)

    # 读取源标注文件
    with open(src_label_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        # 读取图像文件名
        image_name = lines[i].strip()
        i += 1

        # 读取人脸数量
        if i >= len(lines):
            break  # 防止越界
        num_faces = lines[i].strip()
        if not num_faces.isdigit():
            print(f"警告: 跳过无效的人脸数量: {num_faces} (文件: {image_name})")
            i += 1
            continue
        num_faces = int(num_faces)
        i += 1

        # 随机抽样
        if random.random() > sample_ratio:
            i += num_faces  # 跳过未抽中的图像
            continue

        # 构建图像路径
        src_image_path = os.path.join(src_image_dir, image_name)
        if not os.path.exists(src_image_path):
            print(f"警告: 图像文件不存在，跳过: {src_image_path}")
            i += num_faces
            continue

        # 复制图像
        dest_image_path = os.path.join(dest_image_dir, os.path.basename(image_name))
        shutil.copy(src_image_path, dest_image_path)

        # 转换标注文件
        label_file_name = os.path.basename(image_name).replace('.jpg', '.txt')
        dest_label_path = os.path.join(dest_label_dir, label_file_name)

        img = cv2.imread(src_image_path)
        img_height, img_width, _ = img.shape

        with open(dest_label_path, 'w') as label_file:
            for _ in range(num_faces):
                if i >= len(lines):
                    break  # 防止越界
                face_info = lines[i].strip().split()
                if len(face_info) < 4:
                    print(f"警告: 跳过无效的边界框: {face_info} (文件: {image_name})")
                    i += 1
                    continue
                x1, y1, w, h = map(float, face_info[:4])
                x_center = (x1 + w / 2) / img_width
                y_center = (y1 + h / 2) / img_height
                width = w / img_width
                height = h / img_height
                label_file.write(f"0 {x_center} {y_center} {width} {height}\n")
                i += 1

    print(f"抽样完成！图像已保存到 {dest_image_dir}，标注文件已保存到 {dest_label_dir}")
# 示例：处理训练集
sample_images_and_convert_labels(
    src_image_dir='widerface/WIDER_train/images',
    src_label_file='widerface/wider_face_split/wider_face_train_bbx_gt.txt',
    dest_image_dir='widerface/images/train',
    dest_label_dir='widerface/labels/train',
    sample_ratio=0.1
)

# 示例：处理验证集
sample_images_and_convert_labels(
    src_image_dir='widerface/WIDER_val/images',
    src_label_file='widerface/wider_face_split/wider_face_val_bbx_gt.txt',
    dest_image_dir='widerface/images/val',
    dest_label_dir='widerface/labels/val',
    sample_ratio=0.1
)