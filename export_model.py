from ultralytics import YOLO

model = YOLO("yolov8m.pt")
# Używamy quantize=True zamiast przestarzałego half=True
model.export(format="coreml", nms=True)