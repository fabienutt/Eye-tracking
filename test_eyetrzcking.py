import cv2
import mediapipe as mp

# Initialiser les modules MediaPipe pour la détection du visage
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Créer un objet de détection de visage
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Ouvrir la capture vidéo depuis la webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image BGR (par défaut de OpenCV) en RGB pour MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Effectuer la détection des visages
    results = face_detection.process(rgb_frame)

    # Dessiner les repères du visage
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Afficher le résultat
    cv2.imshow('Face Detection', frame)

    # Quitter la fenêtre avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
