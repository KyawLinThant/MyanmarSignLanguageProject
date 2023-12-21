import os
import pickle
import PIL
from PIL import Image
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
# f = r'C:\Users\KyawLinThant\PycharmProjects\ASL_signLanguage\dataset3'
# for file in os.listdir(f):
#     f_img = f+"/"+file
#     img = Image.open(f_img)
#     img = img.resize((400,400))
#     img.save(f_img)

DATA_DIR = './data0to10'

# RESIZE_WIDTH = 640  # Replace with your desired width
# RESIZE_HEIGHT = 480  # Replace with your desired height
# OUTPUT_DIR = './resized_dataset'  # Replace with your desired output directory
#
# # Create the output directory if it doesn't exist
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
# for dir_ in os.listdir(DATA_DIR):
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
#         img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
#
#         # Resize the image
#         img_resized = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))
#
#         # Create the output directory for the current class if it doesn't exist
#         output_class_dir = os.path.join(OUTPUT_DIR, dir_)
#         os.makedirs(output_class_dir, exist_ok=True)
#
#         # Save the resized image to the output directory
#         output_path = os.path.join(output_class_dir, f"resized_{img_path}")
#         cv2.imwrite(output_path, img_resized)
#
#         # Print the path of the saved resized image
#         print(f"Resized image saved to: {output_path}")


data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

f = open('data_final0to10.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()