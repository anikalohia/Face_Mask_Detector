from flask import Flask,render_template,Response,request,jsonify
import cv2,numpy as np
from tensorflow import keras
from keras.models import load_model

app=Flask(__name__)
model = load_model("model/mask_model.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
def detect_face(img):
    coods = face_cascade.detectMultiScale(img)
    return coods


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    file = request.files["frame"]
    img = np.frombuffer(file.read(),np.uint8)
    img = cv2.imdecode(img,cv2.IMREAD_COLOR)
    coods = detect_face(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    if len(coods) == 0:
        return jsonify({"label":"No Face Detected"})
    
    
    x, y, w, h = coods[0]   # take first face
    face = img[y:y+h, x:x+w]   # CROP

    face = cv2.resize(face, (224,224))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)
   
    
    
    pred = model.predict(face)[0][0]
    print(pred)
    
    if pred < 0.5:
        label = "Mask"
    else:
        label = "No Mask"
        
    
    return jsonify({"label":label})
    
if __name__ == "__main__":
    app.run(debug=True)