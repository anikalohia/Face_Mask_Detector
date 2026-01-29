from flask import Flask,render_template,Response,request,jsonify
import cv2,numpy as np
from tensorflow import keras
import os
import gdown
from keras.models import load_model

app=Flask(__name__)
def ensure_model():
    import os, gdown

    os.makedirs("model", exist_ok=True)

    path = "model/mask_detector.h5"

    if not os.path.exists(path):
        gdown.download(
            "https://drive.google.com/file/d/19PVM-GZDALU8ANKxVNSazNupGurB_Rjf/view?usp=sharing",
            path,
            quiet=False
        )

    return path


MODEL_PATH = ensure_model()
model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
def detect_face(gray):
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    return faces

def detect_face_mask(img):
    y_pred = model.predict(img.reshape(1,224,224,3))
    return y_pred[0][0]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files["frame"]
    img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype("float32")

    
    if frame.max() <= 1.0:
        frame *= 255.0

    print(frame.shape, frame.min(), frame.max())

    pred = detect_face_mask(frame)
    

    label = "No Mask" if pred > 0.5 else "Mask"
    return jsonify({"label": label})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)