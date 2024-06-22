import cv2
import os
from pathlib import Path


def Generation(destination, source_folder):
    # Créer le dossier Img_Gen si il n'existe pas déjà
    Directory =destination
    Path(Directory).mkdir(parents=True, exist_ok=True)

    # Liste de toutes les vidéos dans le dossier Vidéos
    chemin = source_folder # Chemin d'accès vers les vidéos
    video_files = [f for f in os.listdir(chemin) if f.endswith(('.mp4', '.avi'))]

    for video_file in video_files:
        
        # Création de l'objet VideoCapture
        cap = cv2.VideoCapture(f'{chemin}/{video_file}') 
        
        # Dossier de destination des images
        destination_folder = f"{Directory}/{Path(video_file).stem}"
        
        # Créer le dossier de destination si il n'existe pas déjà
        Path(destination_folder).mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        while True:
            # Lecture de la frame
            ret, frame = cap.read()
            
            # Si la frame est correctement lue et ret = True
            if not ret:
                break
        
            
            # Save the frame as an image
            img_name = f'{destination_folder}/{Path(video_file).stem}_frame_{frame_count}.jpg'
            cv2.imwrite(img_name, frame)
            
            frame_count += 1

        # Release the VideoCapture object
        cap.release()

    print("Extraction terminée avec succès!")
    

def delete_empty_folders(root):
    deleted = set()
    for current_dir, subdirs, files in os.walk(root, topdown=False):
        still_has_subdirs = False
        for subdir in subdirs:
            if os.path.join(current_dir, subdir) not in deleted:
                still_has_subdirs = True
                break
        if not any(files) and not still_has_subdirs:
            os.rmdir(current_dir)
            deleted.add(current_dir)


Generation("Datasets", "./Vidéos")
delete_empty_folders("Datasets")