import os
import cv2
from ultralytics import YOLO

# Dossier des vidéos
VIDEOS_DIR = os.path.join('.', 'Videos')

# Chemin de la vidéo
video_path = os.path.join(VIDEOS_DIR, 'Vid_Test_100.mp4')
video_path_out = f'{video_path}_out.mp4'

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Fin de la vidéo")

# Récupérez les dimensions d'origine de la vidéo
H, W = frame.shape[:2]

# Dimensions de sortie (utilisez les dimensions d'origine)
size = (W, H)
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), size)

# Chemin du modèle personnalisé
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

# Chargement du modèle personnalisé
model = YOLO(model_path)

threshold = 0.3

# Créez une fenêtre d'affichage
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

while ret:
    results = model(frame)[0]

    # Affichez l'image redimensionnée
    resized_frame = cv2.resize(frame, size)
    cv2.imshow("Detection", resized_frame)  # Affichez la frame redimensionnée
    
    if not ret:
        print("Fin de la vidéo")
        break
    
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            
    cv2.waitKey(1)  # Attend 1 milliseconde pour mettre à jour la fenêtre

    out.write(frame)
    ret, frame = cap.read()

# Libérez les ressources
cap.release()
out.release()
cv2.destroyAllWindows()
