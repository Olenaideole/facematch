import os
import numpy as np
import pandas as pd
import pickle
#import face_recognition #dlib

#yolo
import onnxruntime as ort
import cv2
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings
warnings.filterwarnings('ignore')

#Yolo face detector------------------------------------------------
model_path = "face_detector_m.onnx"
ort_session = ort.InferenceSession(model_path)
# Precompute constants
input_shape = ort_session.get_inputs()[0].shape
INPUT_HEIGHT, INPUT_WIDTH = input_shape[2:]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def inference_yolo(image_path, threshold = 0.2):
    def preprocess_image(image):
        resized_img = cv2.resize(image, (INPUT_HEIGHT, INPUT_WIDTH ), interpolation=cv2.INTER_LINEAR)
        input_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.expand_dims(np.transpose(input_img, (2, 0, 1)), axis=0)

    #image = cv2.imread(image_path)
    image = image_path.copy()
    try:
        img_height, img_width = image.shape[:2]
    except:
        print(image_path , ' is broken')
        return []
    scale_x, scale_y = img_width / INPUT_WIDTH, img_height / INPUT_HEIGHT
    # Preprocess image
    input_img = preprocess_image(image)

    # Perform inference
    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_img})[0][0]
    # Post-process outputs
    filtered_outputs = outputs[outputs[:, 4] > threshold]

    if filtered_outputs.size == 0:
        return []  # No detections above threshold

    # Use argmax directly on the probability column
    # Вот здесь мы выбираем топ 1 предсказание, в котором мы больше всего уверены
    best_detection = filtered_outputs[np.argmax(filtered_outputs[:, 4])]

    # Vectorized scaling of coordinates and clipping
    best_detection[[0, 2]] = np.clip(best_detection[[0, 2]] * scale_x, 0, None)
    best_detection[[1, 3]] = np.clip(best_detection[[1, 3]] * scale_y, 0, None)

    #return best_detection.tolist()
    res_outputs = filtered_outputs.tolist()
    try:
        res = []
        for i in range(len(res_outputs)):
            x1, y1, x2, y2 = res_outputs[i][:4]
            x1, y1, x2, y2 = max(0 , int(x1)), max(0 ,int(y1)), max(0 ,int(x2)), max(0 ,int(y2))
            #return image[y1:y2, x1:x2]
            res.append([x1, y1, x2, y2])
        return res
    except:
        print('broke')
        #face not detected
        return []

#Dlib face detector-------------------------------------------------
'''
def dlib_face_detector(path):
    try:
        image = face_recognition.load_image_file(path)
        face_locations = face_recognition.face_locations(image) #top, right, bottom, left
        #encoding = face_recognition.face_encodings(image)[0]
        converted_cords = [[cord[3], cord[0], cord[1], cord[2]] for cord in face_locations]
        return converted_cords
    except Exception as e:
        # face not_found
        print(e)
        face_locations = [inference_yolo(path)]
    return face_locations

'''
# Face marks------------------------------------------------------------
session = ort.InferenceSession('FaceRegressor.onnx')  # ("face_regressor.onnx")
output_name = session.get_outputs()[0].name
input_name = session.get_inputs()[0].name


# Create a dummy input as an example
# Adjust the shape according to your model's input
def predict_class_path(image_path):  # (image):
        try:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                # normalize the image
                image = image / 255.0
                image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                image = image.astype(np.float32)
                image = np.transpose(image, (2, 0, 1))
                image = np.expand_dims(image, axis=0)

                # Prepare the input dictionary
                input_name = session.get_inputs()[0].name
                input_dict = {input_name: image}

                # Get the name of the output
                output_name = session.get_outputs()[0].name

                # Run inference
                outputs = session.run([output_name], input_dict)

                # Get the maximum class index
                predicted_class = outputs[0][0][0]
                return predicted_class
        except Exception as e:
                print(e)
                return 0


def predict_class_path(image):  # (image):
    try:
        image = cv2.resize(image, (224, 224))
        # normalize the image
        image = image / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        # Prepare the input dictionary
        input_name = session.get_inputs()[0].name
        input_dict = {input_name: image}

        # Get the name of the output
        output_name = session.get_outputs()[0].name

        # Run inference
        outputs = session.run([output_name], input_dict)

        # Get the maximum class index
        predicted_class = outputs[0][0][0]
        return predicted_class
    except Exception as e:
        print(e)
        return 0

def make_df_from_folder(src):
        df = pd.DataFrame(columns=['im_name'])
        i = 0
        # путь к датасету
        for im in os.listdir(src):
                im_name = src + im
                df.loc[i] = [im_name]
                i += 1
        return df


def check_crop_size(x):
        img = cv2.imread(x)
        height, width, channels = img.shape
        if (height < 120) or (width < 120):
                os.remove(x)
                return False
        return True
# Encoding-----------------------------------------------------
'''def dlib_encoding(path):
    try:
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)[0]
    except:
        #print(path) #show images with broken encodings
        encoding = np.zeros(128)

    return encoding'''
from get_face_embeddings import process_image_embeding

#Сравнение---------------------------------------------------------
'''# Находим ближайшего
def threshold_dlib_accuracy(df, i=0 , k=5 ,threshold= 0.033) :
    dlib_face_features_loc = list(df.columns).index('face_features')
    embeddings = np.stack(df.face_features.values).astype('float32')
    index = faiss.IndexFlatL2(128) # фиксированно для длиб
    index.add(embeddings)

    distances, indices = index.search(embeddings[i:i + 1].astype(np.float32), k + 1)  # +1 to include the image itself

    return distances, indices
'''

