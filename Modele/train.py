from ultralytics import YOLO
import pickle

# Load a model
model = YOLO("yolov8n.yaml") # build a new model from scratch


# Utilisation du modèle
results = model.train(data="config.yaml", epochs=1)  # Entrainement du modèle

# Evaluation des performances du modèle
results = model.val()

# Exporter le modèle
# success = model.export(format="onnx", dynamic=True)

# Sauvegardez le modèle dans un fichier
with open('modele.pickle', 'wb') as fichier:
    pickle.dump(model, fichier)