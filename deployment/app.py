import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from PIL import Image
from tensorflow.keras.preprocessing import image


app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)
    
    path = os.path.join(file_path)
    print(path)
    labels={0: 'Cloudy', 1: 'Rain', 2: 'Shine', 3: 'Sunrise'}
    img = image.load_img(path , target_size = (250 , 250))
    img = image.img_to_array(img, dtype=np.uint8)
    img = np.array(img)/255.0
        
    predict = model.predict(img[np.newaxis , ...])
    predicted_class = labels[np.argmax(predict[0] , axis = -1)]
    return render_template('index.html', predict='{}' .format(predicted_class))

if __name__ == '__main__':
    app.run(debug=True)