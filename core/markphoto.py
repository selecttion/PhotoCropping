import sys
import os
import csv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout,
    QWidget, QComboBox, QCheckBox, QMessageBox, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRect, QPoint


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        # self.setAlignment(Qt.AlignCenter)
        self.crop_rect = None
        self.scale_ratio = (1, 1)


    def set_crop_rect(self, rect):
        self.crop_rect = rect
        self.update()

    def set_scale_ratio(self, ratio):
        self.scale_ratio = ratio

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pixmap() or not self.crop_rect:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Step 1: 画半透明遮罩
        overlay = QPixmap(self.size())
        overlay.fill(Qt.transparent)
        overlay_painter = QPainter(overlay)
        overlay_painter.setRenderHint(QPainter.Antialiasing)

        overlay_painter.fillRect(self.rect(), QColor(0, 0, 0, 100))  # 半透明黑色遮罩

        # Step 2: 计算映射后的裁剪框并挖空
        sx, sy = 1 / self.scale_ratio[0], 1 / self.scale_ratio[1]
        scaled_rect = QRect(
            int(self.crop_rect.left() * sx),
            int(self.crop_rect.top() * sy),
            int(self.crop_rect.width() * sx),
            int(self.crop_rect.height() * sy)
        )

        overlay_painter.setCompositionMode(QPainter.CompositionMode_Clear)
        overlay_painter.fillRect(scaled_rect, Qt.transparent)
        overlay_painter.end()

        # Step 3: 显示遮罩
        painter.drawPixmap(0, 0, overlay)

        # Step 4: 绘制红色裁剪框
        pen = QPen(Qt.red, 2, Qt.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(scaled_rect)


class ImageAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像标注工具")
        self.setGeometry(100, 100, 1000, 800)

        self.image_paths = []
        self.current_index = 0
        self.crop_ratios = {1: (7, 10), 2: (2, 3), 3: (3, 4)}
        self.selected_ratio_id = 1
        self.csv_path = ""
        self.dragging = False
        self.scale_ratio = (1, 1)

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        btn_layout = QHBoxLayout()
        self.btn_open_folder = QPushButton("选择图片文件夹")
        self.btn_open_csv = QPushButton("选择输出CSV文件")
        self.btn_prev = QPushButton("上一张")
        self.btn_next = QPushButton("下一张")
        self.btn_save = QPushButton("保存标注")

        for btn in [self.btn_open_folder, self.btn_open_csv, self.btn_prev, self.btn_next, self.btn_save]:
            btn_layout.addWidget(btn)

        layout.addLayout(btn_layout)

        ctrl_layout = QHBoxLayout()
        self.combo_ratio = QComboBox()
        self.combo_ratio.addItems(["10:7", "3:2", "4:3"])
        self.checkbox_padding = QCheckBox("留白")
        self.coord_label = QLabel("坐标：")

        ctrl_layout.addWidget(QLabel("裁剪比例："))
        ctrl_layout.addWidget(self.combo_ratio)
        ctrl_layout.addWidget(self.checkbox_padding)
        ctrl_layout.addWidget(self.coord_label)
        layout.addLayout(ctrl_layout)

        self.image_label = ImageLabel()
        layout.addWidget(self.image_label)

        self.btn_open_folder.clicked.connect(self.open_folder)
        self.btn_open_csv.clicked.connect(self.open_csv)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_save.clicked.connect(self.save_annotation)
        self.combo_ratio.currentIndexChanged.connect(self.change_ratio)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_paths.sort()
            self.current_index = 0
            if self.image_paths:
                self.load_image()

    def open_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "选择CSV文件", filter="CSV Files (*.csv)")
        if path:
            self.csv_path = path

    def load_image(self):
        if not self.image_paths:
            return
        image_path = self.image_paths[self.current_index]
        self.orig_pixmap = QPixmap(image_path)
        self.set_scaled_pixmap()
        self.update_crop_rect()
        self.checkbox_padding.setChecked(False)

    def set_scaled_pixmap(self):
        if hasattr(self, 'orig_pixmap') and not self.orig_pixmap.isNull():
            label_width = self.image_label.width()
            label_height = self.image_label.height()
            self.scaled_pixmap = self.orig_pixmap.scaled(
                label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.scale_ratio = (
                self.orig_pixmap.width() / self.scaled_pixmap.width(),
                self.orig_pixmap.height() / self.scaled_pixmap.height()
            )
            self.image_label.setPixmap(self.scaled_pixmap)
            self.image_label.set_scale_ratio(self.scale_ratio)

            # 确保图片居中
            # self.image_label.setAlignment(Qt.AlignCenter)

    def resizeEvent(self, event):
        self.set_scaled_pixmap()
        self.update_crop_rect()

    def change_ratio(self, index):
        self.selected_ratio_id = index + 1
        self.update_crop_rect()

    def update_crop_rect(self):
        if not hasattr(self, "orig_pixmap") or self.orig_pixmap.isNull():
            return

        ow, oh = self.orig_pixmap.width(), self.orig_pixmap.height()
        rw, rh = self.crop_ratios[self.selected_ratio_id]

        # 修复比例方向：横图使用横向比例，竖图使用竖向比例
        if ow >= oh:
            ratio_w, ratio_h = max(rw, rh), min(rw, rh)
        else:
            ratio_w, ratio_h = min(rw, rh), max(rw, rh)

        # 按比例计算裁剪框尺寸
        if ow / oh >= ratio_w / ratio_h:
            ch = oh
            cw = int(ch * ratio_w / ratio_h)
        else:
            cw = ow
            ch = int(cw * ratio_h / ratio_w)

        x = (ow - cw) // 2
        y = (oh - ch) // 2

        self.crop_rect = QRect(x, y, cw, ch)
        self.coord_label.setText(f"xmin:{x} ymin:{y} xmax:{x + cw} ymax:{y + ch}")
        self.image_label.set_crop_rect(self.crop_rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            mx, my = int(event.x() * self.scale_ratio[0]), int(event.y() * self.scale_ratio[1])
            if self.crop_rect.contains(mx, my):
                self.dragging = True
                self.offset = QPoint(mx, my) - self.crop_rect.topLeft()

    def mouseMoveEvent(self, event):
        if self.dragging:
            mx, my = int(event.x() * self.scale_ratio[0]), int(event.y() * self.scale_ratio[1])
            new_top_left = QPoint(mx, my) - self.offset
            new_crop_rect = QRect(new_top_left, self.crop_rect.size())

            ow, oh = self.orig_pixmap.width(), self.orig_pixmap.height()
            if new_crop_rect.left() < 0:
                new_crop_rect.moveLeft(0)
            if new_crop_rect.top() < 0:
                new_crop_rect.moveTop(0)
            if new_crop_rect.right() > ow:
                new_crop_rect.moveRight(ow)
            if new_crop_rect.bottom() > oh:
                new_crop_rect.moveBottom(oh)

            if self.crop_rect.width() == ow:
                new_crop_rect.moveLeft(0)
            if self.crop_rect.height() == oh:
                new_crop_rect.moveTop(0)

            self.crop_rect = new_crop_rect
            self.coord_label.setText(
                f"坐标：({self.crop_rect.left()}, {self.crop_rect.top()}, "
                f"{self.crop_rect.right()}, {self.crop_rect.bottom()})"
            )
            self.image_label.set_crop_rect(self.crop_rect)

    def mouseReleaseEvent(self, event):
        self.dragging = False

    def save_annotation(self):
        if not self.csv_path:
            QMessageBox.warning(self, "错误", "请先选择CSV文件")
            return

        image_name = os.path.basename(self.image_paths[self.current_index])
        ratio_id = self.selected_ratio_id
        ratio = self.crop_ratios[ratio_id]
        img_w, img_h = self.orig_pixmap.width(), self.orig_pixmap.height()

        if img_w * ratio[1] == img_h * ratio[0]:
            crop_type = 0
            xmin = ymin = xmax = ymax = ""
            QMessageBox.information(self, "提示", f"{image_name} 与选择比例一致，无需裁剪")
        elif self.checkbox_padding.isChecked():
            crop_type = 2
            xmin = ymin = xmax = ymax = ""
        else:
            crop_type = 1
            xmin, ymin, xmax, ymax = self.crop_rect.left(), self.crop_rect.top(), self.crop_rect.right(), self.crop_rect.bottom()

        row = [image_name, crop_type, ratio_id, xmin, ymin, xmax, ymax]
        write_header = not os.path.exists(self.csv_path)

        with open(self.csv_path, "a", newline='', encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Image Name", "Padding", "Aspect Ratio", "xmin", "ymin", "xmax", "ymax"])
            writer.writerow(row)

        QMessageBox.information(self, "已保存", f"{image_name} 标注信息已保存")

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.load_image()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageAnnotator()
    window.show()
    sys.exit(app.exec_())
