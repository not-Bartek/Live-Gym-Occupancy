from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(
    source=0,
    classes=[0],
    show=True,
    stream=True,
    conf=0.4
)

for r in results:
    pass