import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#usage-examples:~:text=of%20each%20category-,Arguments%20for%20YOLO%20Model%20Validation,-When%20validating%20YOLO
# 精度小数点保留位数修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第五点
# 最终论文的参数量和计算量统一以这个脚本运行出来的为准

if __name__ == '__main__':
    model = YOLO('runs/visdrone/yolov6/yolov6l/weights/best.pt') # 选择训练好的权重路径
    model.val(data=r'E:\PycharmProject\002 CV\001 Object Detection\YOLO\YOLOv8v10\ultralytics-20241219\ultralytics-main\ultralytics\cfg\datasets\VisDrone.yaml',
              split='test', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=16,
              # iou=0.7,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val/visdrone',
              name='yolov6/yolov6l',
              )