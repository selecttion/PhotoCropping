冲印的批量图像裁剪小工具，使用YOLOv5+PyQt;
选择输入文件夹，输出文件夹，裁剪比例，裁剪方式后即可使用。
需要在models文件夹下放置自己训练好的人脸检测模型并命名为best


# PhotoCropping-YOLOv5

A PyQt-based image cropping system using YOLOv5 for face and person detection.

## Features
- Face & person detection (YOLOv5)
- Automatic image cropping
- PyQt GUI
- Supports exe packaging (PyInstaller)

## Environment
- Python 3.8+
- PyTorch
- PyQt5

## Notes
Model weights are not included. Please place your own `.pt` files under `models/`.

