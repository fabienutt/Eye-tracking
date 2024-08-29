import cv2
import mediapipe as mp

# Initialiser MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Configurer le détecteur de visages
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Initialiser la capture vidéo (0 pour la webcam par défaut)
cap = cv2.VideoCapture(0)

# Vérifier si la capture vidéo est ouverte correctement
if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la webcam.")
    exit()

while True:
    # Lire une image de la webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Erreur: Impossible de lire une image de la webcam.")
        break

    # Convertir l'image en RGB pour MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Détection des visages
    results = face_detection.process(rgb_frame)
    
    # Convertir l'image RGB de retour en BGR pour OpenCV
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    # Vérifier si des visages ont été détectés
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Dessiner un rectangle autour du visage
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Optionnel : Dessiner les landmarks du visage
            # mp_drawing.draw_detection(frame, detection)
    
    # Afficher l'image avec détection des visages
    cv2.imshow('Face Detection with MediaPipe', frame)
    
    # Quitter si l'utilisateur appuie sur la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture vidéo et fermer toutes les fenêtres
cap.release()
cv2.destroyAllWindows()
