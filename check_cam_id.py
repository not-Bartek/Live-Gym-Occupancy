import cv2

def scan_cameras(max_tested=5):
    available_cams = []
    
    for idx in range(max_tested):
        # cv2.CAP_AVFOUNDATION jest domyślnym backendem na macOS
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cams.append(idx)
                print(f"[OK] Kamera znaleziona pod indeksem: {idx}")
            cap.release()
            
    return available_cams

cams = scan_cameras()
print(f"\nDostępne indeksy kamer: {cams}")

# Wyświetlanie podglądu z każdej kamery po kolei
for idx in cams:
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    print(f"\nWyświetlam kamerę {idx}. Naciśnij 'Spację' lub 'q', aby przejść do kolejnej...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.putText(frame, f"Kamera INDEX: {idx}", (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("Podglad kamer", frame)
        
        key = cv2.waitKey(20) & 0xFF
        if key in (ord(' '), ord('q'), 27):
            break
            
    cap.release()

cv2.destroyAllWindows()