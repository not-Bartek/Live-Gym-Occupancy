import cv2
import time
from ultralytics import YOLO

pending_points = []
active_zone = None

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

def on_mouse_click(event, x, y, flags, param):
    global pending_points, active_zone
    if event == cv2.EVENT_LBUTTONDOWN:
        pending_points.append((x, y))
        
        # Gdy kliknięto 2 punkty, normalizujemy i ustalamy nową strefę
        if len(pending_points) == 2:
            p1, p2 = pending_points
            x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
            y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])
            active_zone = (x1, y1, x2, y2)
            pending_points = []

on_mouse_click.count = 0

# config
model = YOLO("yolo11n.pt")
N_FRAME = 1
n = 5
m = 2
"""
n:
1 - funny moments z kamer cctv
2 - fabryka
3 - chyba kuchnia 
4 - sklep

0 - kamerka
wtedy m to id kamerki
można użyć check_id_cam.py do sprawdzenia id
"""
if n != 0:
    video_path = f"tests/test{n}.mp4"
else:
    video_path = m

cap = cv2.VideoCapture(video_path)

prev_time = 0
frame_idx = -1
annotated_frame = None

# Współrzędne strefy

points = [None, None]


WINDOW_NAME = "CCTV Live Detection"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow("CCTV Live Detection", 1280, 720)

success, frame = cap.read()

#Zaznaczenie punktów
if not success:
    print("Koniec nagrania lub błąd odczytu.")
    exit

on_mouse_click.counter = 0
cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)

while active_zone is None:
    temp_frame = frame.copy()
    if len(pending_points) == 1:
        cv2.circle(temp_frame, pending_points[0], 6, (0, 255, 255), -1)
        draw_text_with_outline(temp_frame, "Kliknij drugi rog strefy...", (20, 50), 0.8, (0, 255, 255))
    else:
        draw_text_with_outline(temp_frame, "Zaznacz strefe (2 klikniecia)", (20, 50), 0.8, (0, 255, 255))

    cv2.imshow(WINDOW_NAME, temp_frame)
    
    key = cv2.waitKey(20) & 0xFF
    if key in (27, ord('q')):
        break

if active_zone is not None:
    temp_frame = frame.copy()
    x1, y1, x2, y2 = active_zone
    cv2.circle(temp_frame, (x1, y1), 6, (0, 255, 255), -1)
    cv2.circle(temp_frame, (x2, y2), 6, (0, 255, 255), -1)
    cv2.imshow(WINDOW_NAME, temp_frame)
    cv2.waitKey(200)
    


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = cap.read()
        if not success:
            print("Błąd odczytu")
            break

    frame_idx += 1
    if frame_idx % N_FRAME != 0:
        continue
    if frame_idx % N_FRAME == 0:
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        results = model.track(frame, classes=[0], conf=0.25, iou=0.5, tracker="bytetrack.yaml",verbose=False)

        # Rysujemy domyślne ramki z YOLO
        annotated_frame = results[0].plot()

        zx1, zy1, zx2, zy2 = active_zone

        # 1. RYSOWANIE STREFY (półprzezroczyste wypełnienie + ramka)
        overlay = annotated_frame.copy()
        if len(pending_points) == 1:
                cv2.circle(annotated_frame, pending_points[0], 6, (0, 255, 255), -1)
        cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
        cv2.rectangle(annotated_frame, (zx1, zy1), (zx2, zy2), (0, 255, 0), 2)

        # 2. ZLICZANIE OSÓB W STREFIE (resetujemy licznik na każdą klatkę)
        people_in_zone = 0
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            foot_x = int((x1 + x2) / 2)
            foot_y = y2
            
            if zx1 <= foot_x <= zx2 and zy1 <= foot_y <= zy2:
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