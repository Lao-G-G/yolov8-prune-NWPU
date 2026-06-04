from ultralytics import YOLO

model = YOLO("weights/yolov8m.pt")
# L1正则的惩罚项系数sr
model.train(
    sr=0,
    data="ultralytics/cfg/datasets/NWPU.yaml",
    epochs=150,
    project='.', 
    name='runs/train-normal',
    batch=24,
    device=0
)