import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v9/yolov9c.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    # model = YOLO('yolo12x.pt')
    model.train(data=r'E:\PycharmProject\002 CV\001 Object Detection\YOLO\ultralytics-main\ultralytics\cfg\datasets\AITOD.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                close_mosaic=0,
                workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                device='0',
                optimizer='SGD', # using SGD
                patience=50, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt,例如YOLO('last.pt')
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/AITOD',
                name='yolov9/yolov9c',
                )