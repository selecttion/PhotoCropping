import sys
from pathlib import Path

# 获取 YOLOv5 目录的绝对路径
YOLOV5_PATH = Path("D:/work/PhotoCropping/yolov5-master").resolve()

# 将 YOLOv5 目录加入 sys.path
if str(YOLOV5_PATH) not in sys.path:
    sys.path.append(str(YOLOV5_PATH))

print(sys.path)

# 尝试导入 DetectMultiBackend
try:
    from models.common import DetectMultiBackend
    print("DetectMultiBackend imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")