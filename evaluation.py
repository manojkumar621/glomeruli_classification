import sys
import os
import numpy as np
import cv2
import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import load_model



# Function to predict and generate evaluation.csv
def generate_predictions(model, image_folder, output_csv='evaluation.csv'):
    model = load_model(model_path)
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    predictions = []

    for img_file in image_files:
      print('image_file', img_file)
      image_height, image_width = 224, 224
      img_path = os.path.join(image_folder, img_file)
      image = cv2.imread(img_path)
      image = cv2.resize(image, (image_height, image_width))
      image = image / 255.0
      image = np.expand_dims(image, axis=0)
      pred = model.predict(image)
      pred_class = 1 if pred[0][0] > 0.5 else 0
      predictions.append({'name': img_file, 'ground truth': pred_class})

    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluation.py <path_to_image_folder>")
        sys.exit(1)

    image_folder = sys.argv[1]
    model_path = 'my_resnet_model_jun11_recall_96.h5'

    # Generate predictions and save to evaluation.csv
    generate_predictions(model_path, image_folder)
