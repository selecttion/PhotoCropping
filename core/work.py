import csv

from PIL import Image
import time
import os
from PyQt5.QtCore import QThread, pyqtSignal
from datetime import datetime
import sys
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern
from typing import List, Tuple


def get_project_root():
    """
    返回项目根目录
    - 源码运行
    - PyInstaller exe
    都能用
    """
    if hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# def load_model(weights_path):
#     # 直接加载本地模型文件
#     model = torch.hub.load('D:\work\PhotoCropping\yolov5', 'custom', path=weights_path, source='local')
#     # 移动模型到指定设备（GPU 或 CPU）
#     model.to(device).eval()  # 设置模型为评估模式
#     return model
def get_project_root():
    if hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PROJECT_ROOT = get_project_root()
YOLOV5_ROOT = os.path.join(PROJECT_ROOT, "yolov5")

if YOLOV5_ROOT not in sys.path:
    sys.path.insert(0, YOLOV5_ROOT)

import torch
from yolov5.models.common import DetectMultiBackend

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weight_name):
    weights_path = os.path.join(PROJECT_ROOT, "models", weight_name)

    model = DetectMultiBackend(
        weights_path,
        device=device,
        dnn=False,
        data=None,
        fp16=False
    )
    model.model.eval()
    return model

face_model = load_model("best.pt")
person_model = load_model("yolov5s.pt")

RATIO_MAP = {1: (7, 10, 1050, 1500), 2: (2, 3, 1200, 1800), 3: (3, 4, 1350, 1800)}
pid = 0


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ 警告：无法读取 {image_path}，尝试使用 PIL 读取！")
        from PIL import Image
        img = Image.open(image_path).convert("RGB")  # 确保是RGB格式
        img = np.array(img)  # 转换为 NumPy 数组
    return img


class ImageData:
    """ 存储单个图像信息 """

    def __init__(self, image_id, image_path):
        self.image_id = image_id  # 图像顺位id
        self.image_path = image_path  # 图像地址
        try:
            with Image.open(image_path) as img:
                self.image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # 读取图像并转换为 OpenCV 格式
                self.width, self.height = img.size  # 直接获取图像尺寸
        except Exception as e:
            print(f"⚠️  警告: 读取图像 {image_path} 失败，错误信息: {e}")
            self.width, self.height = None, None
        self.max_crop_width = None  # 最大化裁剪长宽
        self.max_crop_height = None
        self.need_crop = 0  # 0: 不裁剪, 1: 裁剪, 2: 留白
        self.has_person = False  # 是否有人物
        self.person_count = 0  # 人物数量
        self.subject_coordinates = (0, 0)  # 主体中心坐标
        self.composition_choice = (0, 0)  # 构图选择
        self.face_list = []  # 存储检测到的 face 边界框 (xmin, ymin, xmax, ymax)
        self.person_list = []  # 存储检测到的 person 边界框 (xmin, ymin, xmax, ymax)
        self.crop_coordinates = None  # 裁剪框坐标
        self.loss_values = []  # 图像损失量
        self.image_crop = None  # 裁剪后图像数据
        self.start_time = None  # 开始时间
        self.end_time = None  # 结束时间


class CroppingThread(QThread):
    # 信号
    log_signal = pyqtSignal(str)  # 日志消息信号
    plabel_signal = pyqtSignal(str)  # 更新 label 信号
    finished_signal = pyqtSignal()  # 任务完成信号
    progress_signal = pyqtSignal(int)  # 定义进度信号

    def __init__(self, user_setting):
        super().__init__()
        self.user_setting = user_setting  # 现在 work.py 可以访问 user_setting
        self.is_running = True
        self.image_list = []

    def run(self):
        """ 线程执行的任务 """
        try:
            self.process_cropping()  # 执行裁剪任务
        finally:
            if not self.is_running:
                self.log_signal.emit("————————————————————\n已取消\n————————————————————\n\n")
                return
            if self.user_setting.number:
                self.progress_signal.emit(100)  # 确保任务完成后进度条满格
                # 计算时间
                self.user_setting.end_time = datetime.now()
                all_time_ms = (self.user_setting.end_time - self.user_setting.start_time).total_seconds() * 1000
                average_time_ms = all_time_ms / self.user_setting.number

                self.log_signal.emit("————————————————————")
                self.log_signal.emit(f"🎯 任务已完成！处理{self.user_setting.number}张图片，\n⏱平均用时{average_time_ms:.3f} 毫秒。")
            self.finished_signal.emit()  # 任务完成后发送信号

        # self.process_image_list(self.user_setting.output_path)

    def process_cropping(self):
        global pid
        self.plabel_signal.emit("运行ing")
        print("图像裁剪工作已启动...")
        self.progress_signal.emit(0)  # 重置进度条
        self.user_setting.start_time = datetime.now()

        # 输出文件夹检查
        if not os.path.exists(self.user_setting.input_path):
            try:
                os.makedirs(self.user_setting.output_path)
                self.log_signal.emit("输出文件夹不存在，已自动创建。")
            except Exception as e:
                self.log_signal.emit(f"错误: 无法创建输出文件夹！{str(e)}")
                self.log_signal.emit("已退出")
                return

        # 遍历输入文件夹中的所有图片
        image_extensions = (".png", ".jpg", ".jpeg")
        image_id = 1

        self.log_signal.emit("发现图像: ")
        for root, _, files in os.walk(self.user_setting.input_path):
            for file in files:
                if not self.is_running:
                    return
                if file.lower().endswith(image_extensions):
                    image_path = os.path.join(root, file)

                    # 创建 ImageData 对象
                    image_data = ImageData(image_id, image_path)
                    self.image_list.append(image_data)

                    # 发送日志信号
                    self.log_signal.emit(f"{image_path}")

                    image_id += 1  # 递增 ID

        self.user_setting.number = len(self.image_list)

        if not self.user_setting.number:
            self.log_signal.emit("未找到图像")
            return
        self.log_signal.emit(f"\n发现 {self.user_setting.number} 张图片，准备裁剪...\n")

        self.progress_signal.emit(5)  # 发送进度信号

        # 留白模式
        if self.user_setting.mode == 2:
            for image in self.image_list:
                image.start_time = datetime.now()
                self.add_white_padding(image)
                progress = int((image.image_id / self.user_setting.number * 95) + 5)  # 计算进度百分比
                self.progress_signal.emit(progress)  # 发送进度信号
            self.progress_signal.emit(100)  # 确保任务完成后进度条满格
            return

        # 裁剪模式
        batch_size = 4
        image_batches = [self.image_list[i:i + batch_size] for i in range(0, len(self.image_list), batch_size)]
        for batch in image_batches:
            if not self.is_running:
                break  # 退出循环
            valid_images = []  # 需要进行裁剪的图片
            original_images = []  # 对应的原始图像数据

            # 1. **预处理阶段**
            for image in batch:
                if not self.is_running:
                    break  # 退出循环
                image.start_time = datetime.now()

                ratio_ok, resolution_ok = self.check_ratio_and_resolution(image)
                if ratio_ok and resolution_ok:
                    self.save_image(image=image)

                    pid = pid+1
                    progress = int((pid / self.user_setting.number * 95) + 5)  # 计算进度百分比
                    self.progress_signal.emit(progress)  # 发送进度信号
                    # print(f"第几张图？{pid}")

                    image.need_crop = 0
                    continue

                elif not resolution_ok:
                    self.add_white_padding(image)  # 进行留白

                    pid = pid + 1
                    progress = int((pid / self.user_setting.number * 95) + 5)  # 计算进度百分比
                    self.progress_signal.emit(progress)  # 发送进度信号

                    image.need_crop = 2
                    continue

                self.compute_max_crop_size(image)

                valid_images.append(image)
                original_images.append(image)  # **存储 image 对象和对应的图像数据**
                # self.log_signal.emit(f"{image.image_path}：需要剪")
            if not valid_images:
                continue  # 这一批全是不需要裁剪的，跳过检测步骤

            self.detect_faces_and_persons(original_images)

        print("图像裁剪工作已结束")
        pid = 0
        return

    def stop(self):
        self.is_running = False

    def save_image(self, image=None, new_img=None):
        """ 保存图像到目标文件夹 """
        try:
            if not os.path.exists(self.user_setting.output_path):
                os.makedirs(self.user_setting.output_path)  # 如果输出文件夹不存在，则创建

            # 获取当前时间戳（精确到秒）
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_id = image.image_id  # 直接从 image 获取 ID

            if new_img is not None:
                pil_image = new_img
            elif image is not None:
                pil_image = Image.open(image.image_path)
            else:
                raise ValueError("缺少必要参数：必须提供 image 或 (new_img 和 image)")

            # 生成基础文件名
            base_filename = f"cropped_{image_id}_{timestamp}.jpg"
            save_path = os.path.join(self.user_setting.output_path, base_filename)

            # 处理重名情况
            counter = 1
            while os.path.exists(save_path):
                base_filename = f"cropped_{image_id}_{timestamp}({counter}).jpg"
                save_path = os.path.join(self.user_setting.output_path, base_filename)
                counter += 1

            image.end_time = datetime.now()
            elapsed_time_ms = (image.end_time - image.start_time).total_seconds() * 1000

            # 保存图像
            pil_image.save(save_path, "JPEG", quality=95)  # 以高质量保存

            self.log_signal.emit(f"✅ 图像已保存: {save_path},\n⏱处理用时{elapsed_time_ms:.3f} 毫秒")

        except Exception as e:
            print(f"⚠️ 保存图片失败: {self.user_setting.output_path}, 错误: {e}")

    def check_ratio_and_resolution(self, image_data):
        """检查图像比例、分辨率，并考虑裁剪方向"""
        ratio_key = self.user_setting.ratio
        if ratio_key not in RATIO_MAP:
            return False, False, False  # (比例OK, 分辨率OK, 是否需要裁剪)

        target_h, target_w, min_h, min_w = RATIO_MAP[ratio_key]
        img_w, img_h = image_data.width, image_data.height
        img_ratio = img_w / img_h  # 图片的原始比例

        # 处理裁剪框方向
        if self.user_setting.direction == 1:
            # 原方向：裁剪框跟随图片方向
            if img_w >= img_h:
                target_ratio = target_w / target_h  # 横图，裁剪框也是横的
            else:
                target_ratio = target_h / target_w  # 竖图，裁剪框也是竖的
                min_w, min_h = min_h, min_w
        elif self.user_setting.direction == 2:
            # 强制横向：不管图片如何，裁剪框都宽>高
            target_ratio = max(target_w, target_h) / min(target_w, target_h)
        else:
            # 强制竖向：不管图片如何，裁剪框都高>宽
            target_ratio = min(target_w, target_h) / max(target_w, target_h)
            min_w, min_h = min_h, min_w
        # 计算是否符合比例 & 分辨率
        ratio_ok = abs(img_ratio - target_ratio) < 0.01
        resolution_ok = img_w >= (0.85 * min_w) and img_h >= (0.85 * min_h)

        return ratio_ok, resolution_ok

    def add_white_padding(self, image_data):
        """对分辨率不够的图像加白边"""
        img = Image.open(image_data.image_path)
        img_w, img_h = img.size

        # 目标比例 & 最小宽高
        ratio_key = self.user_setting.ratio
        target_h, target_w, min_h, min_w = RATIO_MAP.get(ratio_key, (7, 10, 1050, 1500))
        target_ratio = target_w / target_h
        original_ratio = img_w / img_h

        print(f"{image_data.image_id}的比例{original_ratio}")

        # 计算新尺寸
        if self.user_setting.direction == 1:  # 原方向
            if original_ratio >= 1:  # 宽长型
                if original_ratio > target_ratio and original_ratio != 1:  # 竖的
                    new_w = int(img_w * 1.1)
                    new_h = int(new_w / target_ratio)
                    print(f"{image_data.image_id}的1.1,{new_w}和{new_h}")
                else:  # 横的
                    new_h = int(img_h * 1.1)
                    new_w = int(new_h * target_ratio)
                    print(f"{image_data.image_id}的1.2,{new_w}和{new_h}")
            else:  # 高长型
                if original_ratio < (1 / target_ratio):
                    new_h = int(img_h * 1.1)
                    new_w = int(new_h * (1 / target_ratio))
                    print(f"{image_data.image_id}的2.1,{new_w}和{new_h}")
                else:
                    new_w = int(img_w * 1.1)
                    new_h = int(new_w / (1 / target_ratio))
                    print(f"{image_data.image_id}的2.2,{new_w}和{new_h}")

        elif self.user_setting.direction == 2:  # 强制横向
            if original_ratio > target_ratio:
                new_w = int(img_w * 1.1)
                new_h = int(new_w / target_ratio)
            else:
                new_h = int(img_h * 1.1)
                new_w = int(new_h * target_ratio)
        else:  # 强制竖向
            if original_ratio < (1 / target_ratio):
                new_h = int(img_h * 1.1)
                new_w = int(new_h * (1 / target_ratio))
            else:
                new_w = int(img_w * 1.1)
                new_h = int(new_w / (1 / target_ratio))

        # 创建白色背景的新图像
        new_img = Image.new("RGB", (new_w, new_h), (255, 255, 255))

        # 计算粘贴位置，保持居中
        paste_x = (new_w - img_w) // 2
        paste_y = (new_h - img_h) // 2
        new_img.paste(img, (paste_x, paste_y))

        self.save_image(image=image_data, new_img=new_img)

    def compute_max_crop_size(self, image_data):
        """计算符合比例的最大裁剪框长宽"""
        img_w, img_h = image_data.width, image_data.height

        # 获取目标比例 & 最小宽高
        ratio_key = self.user_setting.ratio

        target_h, target_w, min_h, min_w = RATIO_MAP[ratio_key]
        target_ratio = target_w / target_h
        original_ratio = img_w / img_h

        if self.user_setting.direction == 1:  # 原方向
            if original_ratio >= 1:  # 宽长型
                if original_ratio > target_ratio:  # 竖的
                    # 图片较宽，限制宽度
                    image_data.max_crop_height = img_h
                    image_data.max_crop_width = int(img_h * target_ratio)
                else:
                    # 图片较高，限制高度
                    image_data.max_crop_width = img_w
                    image_data.max_crop_height = int(img_w / target_ratio)
            else:
                if original_ratio < (1 / target_ratio):
                    image_data.max_crop_width = img_w
                    image_data.max_crop_height = int(img_w / (1 / target_ratio))
                else:
                    image_data.max_crop_height = img_h
                    image_data.max_crop_width = int(img_h * (1 / target_ratio))

        elif self.user_setting.direction == 2:  # 强制横向
            if original_ratio > target_ratio:
                # 图片较宽，限制宽度
                image_data.max_crop_height = img_h
                image_data.max_crop_width = int(img_h * target_ratio)
            else:
                # 图片较高，限制高度
                image_data.max_crop_width = img_w
                image_data.max_crop_height = int(img_w / target_ratio)

        else:  # 强制竖向
            if original_ratio < (1 / target_ratio):
                image_data.max_crop_width = img_w
                image_data.max_crop_height = int(img_w / (1 / target_ratio))
            else:
                image_data.max_crop_height = img_h
                image_data.max_crop_width = int(img_h * (1 / target_ratio))

        print(f"{image_data.image_id}最大裁剪尺寸：{image_data.max_crop_width}x{image_data.max_crop_height}")

    def calculate_composition_choice(self, image_data):
        """ 计算目标点在构图三分线上最接近的位置 """

        img_w, img_h = image_data.width, image_data.height
        target_x, target_y = image_data.subject_coordinates
        # 计算 x 方向上的三分线
        thirds_x = [img_w / 3, img_w / 2, 2 * img_w / 3]  # [1/3线, 中点, 2/3线]

        # 计算 y 方向上的三分线
        thirds_y = [img_h / 3, img_h / 2, 2 * img_h / 3]  # [1/3线, 中点, 2/3线]

        # 计算 target_x 到三分线的距离，选取最近的
        distances_x = [abs(target_x - pos) for pos in thirds_x]
        closest_x = distances_x.index(min(distances_x)) + 1  # +1 是因为索引 0 对应 1/3 线，索引 1 对应 1/2 线

        # 计算 target_y 到三分线的距离，选取最近的
        distances_y = [abs(target_y - pos) for pos in thirds_y]
        closest_y = distances_y.index(min(distances_y)) + 1  # +1 是因为索引 0 对应 1/3 线，索引 1 对应 1/2 线

        image_data.composition_choice = (closest_x, closest_y)
        return closest_x, closest_y

    def crop_image_with_composition(self, image_data):
        """
        根据构图选择裁剪图片，并使裁剪框尽量与目标点对齐。

        参数：
            image_data: 图片对象，包含路径、ID、尺寸、构图选择、最大裁剪尺寸等信息。
            target_point: (x, y) 坐标，裁剪框应尽量与该点对齐。
        """
        # 加载图像
        img = Image.open(image_data.image_path)
        img_w, img_h = img.size

        # 获取裁剪框尺寸
        crop_w, crop_h = image_data.max_crop_width, image_data.max_crop_height
        target_x, target_y = image_data.subject_coordinates  # 目标点坐标

        # 获取构图偏移量
        composition_x_map = {0: 0, 1: 1 / 3, 2: 1 / 2, 3: 2 / 3}
        composition_y_map = {0: 0, 1: 1 / 3, 2: 1 / 2, 3: 2 / 3}

        offset_x = composition_x_map.get(image_data.composition_choice[0], 0)
        offset_y = composition_y_map.get(image_data.composition_choice[1], 0)

        # 计算裁剪框左上角坐标
        crop_x = int(target_x - offset_x * crop_w)
        crop_y = int(target_y - offset_y * crop_h)

        # **边界检查，确保裁剪框不会超出原图**
        crop_x = max(0, min(crop_x, img_w - crop_w))
        crop_y = max(0, min(crop_y, img_h - crop_h))

        image_data.crop_coordinates = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
        print("裁剪：", image_data.crop_coordinates)

        # 裁剪图片
        cropped_img = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        image_data.image_crop = cropped_img
        self.save_image(image=image_data, new_img=cropped_img)

        # 损失量评估
        # self.compute_ssim_rgb(image_data)

    def detect_faces_and_persons(self, image_data_pairs):
        """ 批量检测图片中的人脸和人体，并存储结果到 image 结构体 """
        global pid
        img_list = []
        image_dict = {}  # 存储 image_id -> image 结构体的映射

        # **1. 批量预处理**
        for image in image_data_pairs:
            image_dict[image.image_id] = image  # 按 image_id 存入字典
            # print(f"加{image.image_path}进去")

            img_list.append(image.image_path)

        if not img_list:
            return []  # 避免空批量报错

        print(f"输入的图像数量: {len(img_list)}")
        # **2. 进行批量检测**
        face_results = face_model(img_list)
        person_results = person_model(img_list)

        # print(f"face_results 列表的长度: {len(face_results)}")
        # print(f"face_results 列表: {face_results}")

        for i in range(len(image_dict)):  # 遍历每一张图片的检测结果
            if not self.is_running:
                break  # 退出循环
            image_id = list(image_dict.keys())[i]  # 获取 image_id
            image = image_dict[image_id]  # 通过 image_id 获取对应的 image 结构体
            print(f"处理图片：{image_id}")

            # 发送进度信号
            progress = int(pid / self.user_setting.number * 95 + (95 / self.user_setting.number * 0.3) + 5)
            self.progress_signal.emit(progress)

            # 获取当前图片对应的人脸检测结果
            face_det = face_results.xyxy[i].clone()  # 克隆张量，避免 in-place 修改错误
            person_det = person_results.xyxy[i].clone()  # 获取当前图片的人体检测结果
            # print(face_det)
            # print(person_det)

            # 先将 face_det 的 class_id 转换为 int
            face_det[:, 5] = face_det[:, 5].int()

            # 解析 face_det：筛选 class_id=0 且置信度 > 0.4
            face_mask = (face_det[:, 5] == 0) & (face_det[:, 4] > 0.4)
            face_filtered = face_det[face_mask]  # 取符合条件的框

            if len(face_filtered) > 0:
                image.face_list = face_filtered[:, :4].int().tolist()  # 转换为 int 并存入 [x1, y1, x2, y2, class]
            else:
                image.face_list = []

            # 解析 person_det：先将 class_id 转换为 int
            person_det[:, 5] = person_det[:, 5].int()
            person_mask = person_det[:, 5] == 0
            persons_filtered = person_det[person_mask]

            if len(persons_filtered) > 0:
                # 计算每个框的面积 (width * height)
                areas = (persons_filtered[:, 2] - persons_filtered[:, 0]) * (
                        persons_filtered[:, 3] - persons_filtered[:, 1])

                # 计算原图像面积
                img_area = image.width * image.height

                # 筛选出面积大于原图像 1% 的检测框
                valid_mask = areas > (0.01 * img_area)
                persons_valid = persons_filtered[valid_mask]

                if len(persons_valid) > 0:
                    image.person_list = persons_valid[:, :4].int().tolist()  # 取 x1, y1, x2, y2 并转换为 int
                    image.has_person = True
                else:
                    image.person_list = []
            else:
                image.person_list = []  # 没有符合条件的框，置空

            # print(f"有人吗？{image.has_person}")
            # print(f"人：{image.person_list}")
            #
            # # 调用绘制框的函数
            # image_with_boxes = self.draw_boxes(image)
            #
            # if image_with_boxes is not None:
            #     # 调整图像尺寸，使其适合屏幕
            #     image_resized = self.resize_image(image_with_boxes)
            #
            #     # 显示缩放后的图像
            #     cv2.imshow('Detection Preview', image_resized)
            #     cv2.waitKey(0)  # 等待按键输入
            #     cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口

            self.select_subject_and_crop(image)

        return list(image_dict.values())  # 返回更新后的 image 结构体列表

    def resize_image(self, image, max_width=1600, max_height=1000):
        """
        将图像缩放到适应屏幕的大小
        :param image: 输入的图像
        :param max_width: 图像的最大宽度
        :param max_height: 图像的最大高度
        :return: 缩放后的图像
        """
        height, width = image.shape[:2]

        # 如果图像尺寸已经小于最大尺寸，则不进行缩放
        if width <= max_width and height <= max_height:
            return image

        # 计算缩放比例
        scale_width = max_width / width
        scale_height = max_height / height
        scale = min(scale_width, scale_height)

        # 缩放图像
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))

        return resized_image

    def draw_boxes(self, image_data):
        """
        绘制人脸和人物框
        :param image_data: 包含图像和检测框的 ImageData 对象
        :return: 绘制了框的图像
        """
        if image_data.image is None:
            print(f"⚠️ 图像 {image_data.image_path} 加载失败，无法绘制框！")
            return None

        image = image_data.image  # 获取图像

        # 绘制人脸框
        for box in image_data.face_list:
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 140, 0), 5)  # 蓝色框

        # 绘制人体框
        for box in image_data.person_list:
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 7)  # 红色框

        return image

    def select_subject_and_crop(self, image_data):
        """
        根据 face_list 和 person_list 选择主体物，并执行裁剪或留白操作
        """
        global pid

        # 获取图像中心点坐标
        center_x, center_y = image_data.width / 2, image_data.height / 2
        image_data.composition_choice = (2, 2)  # 暂时固定为 (2,2)

        # 如果没有检测到人体，则直接返回
        if not image_data.has_person:
            image_data.subject_coordinates = (center_x, center_y)
            self.crop_image_with_composition(image_data)
            return

        def calculate_weighted_score(box, face_boxes=None):
            """
            计算目标框的加权评分：
            - 对于 person 框，基于 **最靠近中心点**（50%）+ **面积最大**（50%）。
            - 对于包含 face 的 person 框，基于 **最靠近中心点**（50%）+ **face 面积最大**（50%）。
            """

            # 解析框坐标
            x_min, y_min, x_max, y_max = box
            box_center_x, box_center_y = (x_min + x_max) / 2, (y_min + y_max) / 2  # 计算框的中心点
            box_area = (x_max - x_min) * (y_max - y_min)  # 计算框的面积

            # 计算框与图像中心点的欧几里得距离
            distance_score = 1 / (
                        np.linalg.norm(np.array([box_center_x, box_center_y]) - np.array([center_x, center_y])) + 1e-6)

            # 计算面积占比得分
            area_score = box_area / (image_data.width * image_data.height)

            # 如果提供了 face_boxes，则计算 face 相关的得分
            if face_boxes:
                face_overlap_scores = []  # 存储每个 face 的重叠得分
                for fx_min, fy_min, fx_max, fy_max in face_boxes:
                    face_area = (fx_max - fx_min) * (fy_max - fy_min)  # 计算 face 框的面积
                    overlap_x_min, overlap_y_min = max(x_min, fx_min), max(y_min, fy_min)  # 计算交集区域
                    overlap_x_max, overlap_y_max = min(x_max, fx_max), min(y_max, fy_max)
                    overlap_area = max(0, overlap_x_max - overlap_x_min) * max(0, overlap_y_max - overlap_y_min)

                    # 计算 face 被包含的比例
                    overlap_ratio = overlap_area / face_area
                    face_overlap_scores.append(overlap_ratio)

                face_score = max(face_overlap_scores) if face_overlap_scores else 0  # 选取最大 face 覆盖率
                return 0.4 * distance_score + 0.6 * area_score + 0.3 * face_score  # 计算最终得分
            else:
                return 0.5 * distance_score + 0.7 * area_score  # 仅基于中心点和面积计算得分

        def is_face_inside_person(face_box, person_box, overlap_threshold=0.4):
            """
            判断 face_box 是否被 person_box 包含或大部分重合
            - overlap_threshold: face 框与 person 框的重叠比例（默认为 40%）
            """
            fx_min, fy_min, fx_max, fy_max = face_box
            px_min, py_min, px_max, py_max = person_box

            # 计算 face 框的面积
            face_area = (fx_max - fx_min) * (fy_max - fy_min)

            # 计算交集区域
            overlap_x_min = max(fx_min, px_min)
            overlap_y_min = max(fy_min, py_min)
            overlap_x_max = min(fx_max, px_max)
            overlap_y_max = min(fy_max, py_max)

            overlap_width = max(0, overlap_x_max - overlap_x_min)
            overlap_height = max(0, overlap_y_max - overlap_y_min)
            overlap_area = overlap_width * overlap_height

            # 计算 face 框与 person 框的重叠比例
            overlap_ratio = overlap_area / face_area

            return overlap_ratio >= overlap_threshold

        # **情况 1**：没有检测到人脸，只有 person 框
        if not image_data.face_list:
            # 计算所有 person 框的得分
            scores = [calculate_weighted_score(person_box) for person_box in image_data.person_list]
            best_idx = np.argmax(scores)  # 选取得分最高的框
            image_data.person_list = [image_data.person_list[best_idx]]  # 仅保留主体物 person 框

            # 计算主体物的中心点坐标
            x_min, y_min, x_max, y_max = image_data.person_list[0]
            image_data.subject_coordinates = ((x_min + x_max) / 2, (y_min + y_max) / 2)

        # **情况 2**：检测到人脸，选择最匹配的 person 框
        else:
            # 计算所有 person 框的得分（考虑 face 的因素）
            scores = [calculate_weighted_score(person_box, image_data.face_list) for person_box in
                      image_data.person_list]
            print(f"{image_data.image_id}的分{scores}")
            best_idx = np.argmax(scores)  # 选取得分最高的框

            # 找出得分相近的 person 框（相差小于）
            best_score = scores[best_idx]
            similar_boxes = [box for i, box in enumerate(image_data.person_list) if abs(scores[i] - best_score) < 0.045]

            # 仅保留相近得分的主体物框
            image_data.person_list = similar_boxes

            # 筛选 face_list
            filtered_faces = []
            for face_box in image_data.face_list:
                # 只保留至少有一个 person 框包含或大部分重合的 face
                if any(is_face_inside_person(face_box, person_box) for person_box in image_data.person_list):
                    filtered_faces.append(face_box)

            image_data.face_list = filtered_faces  # 更新 face_list

            # 计算主体物的中心点坐标
            if len(image_data.face_list) == 0 :
                x_min, y_min, x_max, y_max = image_data.person_list[0]
                image_data.subject_coordinates = ((x_min + x_max) / 2, (y_min + y_max) / 2)
            elif len(image_data.face_list) == 1 :
                # 仅有一个 face，计算其主体坐标
                fx_min, fy_min, fx_max, fy_max = image_data.face_list[0]
                face_x = (fx_min + fx_max) / 2
                face_y = fy_min + (fy_max - fy_min) / 2  # 取 face 框的 1/2 处
                image_data.subject_coordinates = (face_x, face_y)
            else:
                # 多个 face，计算所有 face 坐标的平均值
                face_coords = []
                for fx_min, fy_min, fx_max, fy_max in image_data.face_list:
                    face_x = (fx_min + fx_max) / 2
                    face_y = fy_min + (fy_max - fy_min) / 2
                    face_coords.append((face_x, face_y))
                image_data.subject_coordinates = tuple(np.mean(face_coords, axis=0))

        # **执行裁剪**

        # # 调用绘制框的函数
        # image_with_boxes = self.draw_boxes(image_data)
        # if image_with_boxes is not None:
        #     # 调整图像尺寸，使其适合屏幕
        #     image_resized = self.resize_image(image_with_boxes)
        #
        #     # 显示缩放后的图像
        #     cv2.imshow('Detection Preview', image_resized)
        #     cv2.waitKey(0)  # 等待按键输入
        #     cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口

        # 计算构图选择
        self.calculate_composition_choice(image_data)
        # 发送进度信号
        progress = int(pid / self.user_setting.number * 95 + (95 / self.user_setting.number * 0.6) + 5)
        self.progress_signal.emit(progress)

        if image_data.face_list:
            face_x_min = min(face[0] for face in image_data.face_list)  # 最左侧边界
            face_y_min = min(face[1] for face in image_data.face_list)  # 最上侧边界
            face_x_max = max(face[2] for face in image_data.face_list)  # 最右侧边界
            face_y_max = max(face[3] for face in image_data.face_list)  # 最下侧边界

            face_width = face_x_max - face_x_min  # face 整体区域宽度
            face_height = face_y_max - face_y_min  # face 整体区域高度

            # **检查是否可以裁剪**
            if face_width <= image_data.max_crop_width and face_height <= image_data.max_crop_height:
                # **可以裁剪**
                image_data.need_crop = 1
                self.crop_image_with_composition(image_data)
            else:
                # **face 框太大，避免裁剪影响 face，改为留白**
                image_data.need_crop = 2
                self.add_white_padding(image_data)
        else:
            # **没有 face 框，直接裁剪**
            image_data.need_crop = 1
            self.crop_image_with_composition(image_data)

        # 发送进度信号
        pid = pid+1
        progress = int((pid / self.user_setting.number * 95) + 5)
        self.progress_signal.emit(progress)
        # print(f"第几张图？{pid}")

    def compute_ssim_rgb(self, image_data):
        """
            SSIM损失量
        """
        image_data.image_crop = cv2.cvtColor(np.array(image_data.image_crop), cv2.COLOR_RGB2BGR)
        # 确保图像大小一致
        if image_data.image.shape != image_data.image_crop.shape:
            resized_crop = cv2.resize(image_data.image_crop, (image_data.image.shape[1], image_data.image.shape[0]))
        else:
            resized_crop = image_data.image_crop

        # 初始化变量
        scores = []

        # 分别计算每个通道的SSIM
        for i in range(3):  # BGR 顺序
            score, _ = ssim(image_data.image[:, :, i], resized_crop[:, :, i], full=True)
            scores.append(score)

        avg_score = sum(scores) / 3
        print(f"RGB通道平均 SSIM: {avg_score:.4f}")

    # 损失量
    def compute_crop_loss(self, image: np.ndarray,
                          crop_coords: Tuple[int, int, int, int],
                          person_list: List[Tuple[int, int, int, int]]) -> dict:
        """
        计算裁剪损失：颜色、纹理、人像损失
        """
        xmin, ymin, xmax, ymax = crop_coords
        crop = image[ymin:ymax, xmin:xmax]

        # 颜色损失
        def color_hist_loss(img1, img2):
            hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
            hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            return 1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        color_loss = color_hist_loss(image, crop)

        # 纹理损失
        def texture_hist_loss(img1, img2):
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            lbp1 = local_binary_pattern(gray1, P=8, R=1.0)
            lbp2 = local_binary_pattern(gray2, P=8, R=1.0)
            hist1, _ = np.histogram(lbp1.ravel(), bins=256, range=(0, 256), density=True)
            hist2, _ = np.histogram(lbp2.ravel(), bins=256, range=(0, 256), density=True)
            return np.sum(np.abs(hist1 - hist2)) / 2

        texture_loss = texture_hist_loss(image, crop)

        # person损失
        def compute_person_loss(person_list, crop_box):
            crop_xmin, crop_ymin, crop_xmax, crop_ymax = crop_box
            crop_area = (crop_xmax - crop_xmin) * (crop_ymax - crop_ymin)
            total_person_area = 0
            cropped_person_area = 0
            for (xmin, ymin, xmax, ymax) in person_list:
                area = (xmax - xmin) * (ymax - ymin)
                total_person_area += area
                inter_xmin = max(xmin, crop_xmin)
                inter_ymin = max(ymin, crop_ymin)
                inter_xmax = min(xmax, crop_xmax)
                inter_ymax = min(ymax, crop_ymax)
                if inter_xmin < inter_xmax and inter_ymin < inter_ymax:
                    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
                    cropped_person_area += inter_area
            if total_person_area == 0:
                return 0
            return 1 - cropped_person_area / total_person_area

        person_loss = compute_person_loss(person_list, crop_coords)

        return {
            "color_loss": color_loss,
            "texture_loss": texture_loss,
            "person_loss": person_loss
        }

    def visualize_crop_and_loss(self, image: np.ndarray,
                                crop_coords: Tuple[int, int, int, int],
                                person_list: List[Tuple[int, int, int, int]],face_list: List[Tuple[int, int, int, int]],
                                loss_dict: dict,
                                window_name: str = "Crop Loss Visualization",
                                save_path: str = None):
        vis_img = image.copy()
        xmin, ymin, xmax, ymax = crop_coords

        # ✅ 裁剪区域框（红色，加粗）
        cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=15)

        # ✅ person框（绿色，加粗）
        for (pxmin, pymin, pxmax, pymax) in person_list:
            cv2.rectangle(vis_img, (pxmin, pymin), (pxmax, pymax), (0, 255, 0), thickness=7)
        # 脸
        for (fxmin, fymin, fxmax, fymax) in face_list:
            cv2.rectangle(vis_img, (fxmin, fymin), (fxmax, fymax), (255, 140, 0), thickness=7)

        # ✅ 放大字体大小和加粗
        info_text = f"Color: {loss_dict['color_loss']:.3f} | Texture: {loss_dict['texture_loss']:.3f} | Person: {loss_dict['person_loss']:.3f}"
        font_scale = 3
        font_thickness = 5
        cv2.putText(vis_img, info_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # ✅ 缩放显示（防止太大）
        max_width = 1280
        max_height = 1280
        h, w = vis_img.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_img = cv2.resize(vis_img, (new_w, new_h))

        # ✅ 显示图像
        cv2.imshow(window_name, resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_image_list(self, save_dir):
        # 设置 CSV 文件的路径
        csv_file_path = save_dir + '/output.csv' if save_dir else 'output.csv'

        # 打开 CSV 文件进行写入
        with open(csv_file_path, mode='w', newline='') as csvfile:
            fieldnames = ["Image Name", "Padding", "Aspect Ratio", "xmin", "ymin", "xmax", "ymax"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # 写入表头
            writer.writeheader()


            for idx, image_obj in enumerate(self.image_list):
                # 提取相关信息
                image_name = os.path.basename(image_obj.image_path)  # 获取图片的文件名
                padding = image_obj.need_crop
                aspect_ratio = self.user_setting.ratio  # 假设 `self.user_setting.ratio` 是比例设置
                if image_obj.crop_coordinates:
                    xmin, ymin, xmax, ymax = image_obj.crop_coordinates
                else:
                    xmin = ymin = xmax = ymax = ""
                # 将每行数据写入 CSV 文件
                writer.writerow({
                    "Image Name": image_name,
                    "Padding": padding,
                    "Aspect Ratio": aspect_ratio,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                })

                if not image_obj.need_crop == 1:
                    continue

                # print("图片id：", image_obj.image_id)
                # image_np = image_obj.image
                # crop_coords = image_obj.crop_coordinates
                # person_boxes = image_obj.person_list
                # face_boxes = image_obj.face_list
                # loss = self.compute_crop_loss(image_np, crop_coords, person_boxes)
                # print("损失：", loss)
                #
                #
                # self.visualize_crop_and_loss(
                #     image=image_np,
                #     crop_coords=crop_coords,
                #     person_list=person_boxes,
                #     face_list=face_boxes,
                #     loss_dict=loss,
                #     window_name=f"Image {idx + 1} Loss Visualization",
                # )

