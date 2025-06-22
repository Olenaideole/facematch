import cv2
import numpy as np
import onnxruntime as ort

# === Global session and model configuration ===
SESSION = ort.InferenceSession("./RegFaceBest.onnx")
INPUT_NAME = SESSION.get_inputs()[0].name
OUTPUT_NAME = SESSION.get_outputs()[0].name
INPUT_SIZE = SESSION.get_inputs()[0].shape[2:]
print(INPUT_SIZE)

# === Precomputed normalization constants ===
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
INV_STD = 1 / STD
MEAN_INV_STD = -MEAN * INV_STD
SCALE = INV_STD / 255.0

# === Image processing and inference ===
def process_image_path(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INPUT_SIZE[1], INPUT_SIZE[0]))
    image = image.astype(np.float32)
    image = image * SCALE + MEAN_INV_STD
    image = image.transpose(2, 0, 1)[np.newaxis, :]

    outputs = SESSION.run([OUTPUT_NAME], {INPUT_NAME: image})
    return outputs[0][0]

def process_image_embeding(image):
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INPUT_SIZE[1], INPUT_SIZE[0]))
    image = image.astype(np.float32)
    image = image * SCALE + MEAN_INV_STD
    image = image.transpose(2, 0, 1)[np.newaxis, :]

    outputs = SESSION.run([OUTPUT_NAME], {INPUT_NAME: image})
    return outputs[0][0]

if __name__ == "__main__":
    image_path = 'faces/img.png'
    embeddings = process_image_path(image_path)
    print(f"Embeddings shape: {embeddings.shape}")
    #print(f"Embeddings: {embeddings}")
