import cv2
import time
from ultralytics import YOLO

def draw_text_with_outline(img, text, pos=(20, 50), font_scale=0.8, 
                           text_color=(170, 100, 240), outline_color=(0, 0, 0), 
                           thickness=2, outline_width=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos

    # Rysujemy kontur dookoła (8 kierunków)
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                cv2.putText(img, text, (x + dx, y + dy), font, font_scale, 
                            outline_color, thickness, cv2.LINE_AA)

    # Rysujemy główny kolor na środku
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

# config
model = YOLO("yolo11n.pt")
N_FRAME = 1
n = 4
# video_path = f"tests/test{n}.mp4"
video_path = 3

cap = cv2.VideoCapture(video_path)

prev_time = 0
frame_idx = -1
annotated_frame = None

# Współrzędne strefy
ZONE_X1, ZONE_Y1 = 150, 800
ZONE_X2, ZONE_Y2 = 1300, 1500

cv2.namedWindow("CCTV Live Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("CCTV Live Detection", 1280, 720)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Koniec nagrania lub błąd odczytu.")
        break

    frame_idx += 1

    if frame_idx % N_FRAME == 0:
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        results = model(frame, classes=[0], conf=0.25, iou=0.5, verbose=False)

        # Rysujemy domyślne ramki z YOLO
        annotated_frame = results[0].plot()

        # 1. RYSOWANIE STREFY (półprzezroczyste wypełnienie + ramka)
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
        cv2.rectangle(annotated_frame, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (0, 255, 0), 2)

        # 2. ZLICZANIE OSÓB W STREFIE (resetujemy licznik na każdą klatkę)
        people_in_zone = 0
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            foot_x = int((x1 + x2) / 2)
            foot_y = y2
            
            if ZONE_X1 <= foot_x <= ZONE_X2 and ZONE_Y1 <= foot_y <= ZONE_Y2:
                people_in_zone += 1
                color = (0, 255, 0)  # Zielony: w strefie
            else:
                color = (0, 0, 255)  # Czerwony: poza strefą

            cv2.circle(annotated_frame, (foot_x, foot_y), 5, color, -1)

        # 3. NAPISY NA EKRANIE
        people_count = len(results[0].boxes)

        draw_text_with_outline(annotated_frame, f"Osoby w kadrze: {people_count}",
                                (10, 30), 0.8, (170, 100, 240), (0, 0, 0), 2)
        
        draw_text_with_outline(annotated_frame, f"FPS: {int(fps)}", 
                                (10, 60), 0.8, (200, 255, 130), (40, 40, 40), 2)

        # Nowy napis: dokładnie pod FPS (y=90)
        draw_text_with_outline(annotated_frame, f"Osoby w strefie: {people_in_zone}", 
                                (10, 90), 0.8, (0, 255, 150), (0, 0, 0), 2)

    if annotated_frame is not None:
        cv2.imshow("CCTV Live Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()