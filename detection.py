import cv2
import time
from ultralytics import YOLO

#config
model = YOLO("yolo11m.pt")
N_FRAME = 10
n=2
video_path = f"tests/test{n}.mp4"


cap = cv2.VideoCapture(video_path)

prev_time = 0

cv2.namedWindow("CCTV Live Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("CCTV Live Detection", 1280, 720)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Koniec nagrania lub błąd odczytu.")
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    results = model(frame, classes=[0], conf=0.15, iou=0.6, verbose=False)

    annotated_frame = results[0].plot()

    people_count = len(results[0].boxes)

    cv2.putText(
        annotated_frame, f"Osoby w kadrze: {people_count}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
    )
    cv2.putText(
        annotated_frame, f"FPS: {int(fps)}", (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
    )

    cv2.imshow("CCTV Live Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()