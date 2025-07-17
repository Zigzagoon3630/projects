import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import json
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

#загрузка модели для предсказания целостности стен
model = tf.keras.models.load_model('model/model_vgg.h5')

def predict_image(image_path):
    try:
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        predict = model.predict(img_array)[0][0]
        return {
            "damage_probability": float(predict) * 100 ,  
            "image_url": image_path
        }
    except Exception as e:
        print(f"Ошибка при анализе изображения: {e}")
        return {
            "damage_probability": 0.0,
            "image_url": image_path
        }

@app.route('/')
def index():
    return render_template('index1_buildings.html')

#добавление возможности отправлять свои изображения на сайт для предсказания их целостности
@app.route('/process', methods=['POST'])
def process():
    if 'objectId' in request.form:
        object_id = request.form.get('objectId')
        return redirect(url_for('show_results', id=object_id))
    
    elif 'image' in request.files:
        file = request.files['image']
        if file.filename == '':
            return redirect(url_for('index'))
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        prediction = predict_image(filepath)
        return redirect(url_for('show_custom_result', prediction=json.dumps(prediction)))
    
    return redirect(url_for('index'))

@app.route('/results/<id>')
def show_results(id):
    try:
        with open(f'data/building_{id}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return render_template('index2_buildings.html', data=data, id=id)
    except FileNotFoundError:
        return "Объект не найден", 404

@app.route('/details/<id>/<int:index>')
def show_details(id, index):
    try:
        with open(f'data/building_{id}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return render_template('index3_buildings.html', data=data, index=index)
    except FileNotFoundError:
        return "Объект не найден", 404

@app.route('/custom-result')
def show_custom_result():
    prediction = json.loads(request.args.get('prediction', '{}'))
    return render_template('index3_buildings.html', prediction=prediction)

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)

@app.route('/static/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
