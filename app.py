import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)


model = load_model('efficientNet.keras')
print('Model yüklendi. http://127.0.0.1:5000/')


def tahmin_et(mr):
    try:
        opencvImage = cv2.cvtColor(np.array(mr), cv2.COLOR_RGB2BGR)
        mr = cv2.resize(opencvImage, (150, 150))
        mr = mr.reshape(1, 150, 150, 3)
        p = model.predict(mr)
        p = np.argmax(p, axis=1)[0]

        if p == 0:
            p = 'Glioma Tümörü'
        elif p == 1:
            return 'Model, tümör olmadığını öngörüyor.'
        elif p == 2:
            p = 'Meninjiyom Tümörü'
        else:
            p = 'Hipofiz Tümörü'

        if p != 1:
            return f'Model bunun bir {p} olduğunu öngörüyor.'
    except Exception as e:
        return f'HATA!! : {str(e)}'


def getResult(img):
    mr = cv2.imread(img)
    mr = Image.fromarray(mr, 'RGB')
    mr = mr.resize((64, 64))
    mr = np.array(mr)
    input_img = np.expand_dims(mr, axis=0)
    sonuc=model.predict_step(input_img)
    return sonuc


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        anaYol = os.path.dirname(__file__)
        dosyYolu = os.path.join(
            anaYol, 'uploads', secure_filename(f.filename))
        f.save(dosyYolu)
        value=getResult(dosyYolu)
        sonuc=tahmin_et(value) 
        return sonuc
    return None


if __name__ == '__main__':
    app.run(debug=True)