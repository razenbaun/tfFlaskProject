import os
import pickle
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify
from matrix import get_matrix_iris, get_matrix_beloved_color
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# print(tf.keras.__version__)

app = Flask(__name__)

menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"},
        {"name": "Лаба 4", "url": "p_lab4"},
        {"name": "нейрон", "url": "p_lab5"},
        {"name": "одежда", "url": "p_lab6"}
        ]

model_class = tf.keras.models.load_model('model/classification_model.h5')
model_weather = load_model('model/weather_model.h5')


loaded_model_knn = pickle.load(open('model/Iris_pickle_file_knn', 'rb'))
loaded_model_linel = pickle.load(open('model/Iris_pickle_file_jilie', 'rb'))
loaded_model_logic = pickle.load(open('model/Iris_pickle_file_logic', 'rb'))
loaded_model_tree = pickle.load(open('model/Iris_pickle_file_tree', 'rb'))

get_matrix_beloved_color(loaded_model_logic, 'logic')
get_matrix_iris(loaded_model_knn, 'knn')
get_matrix_beloved_color(loaded_model_tree, 'tree')


@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные Петрачков Александр ПрИ-201",
                           menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = loaded_model_knn.predict(X_new)
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model="Это " + pred[0])


@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Линейная регрессия", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1'])]])
        pred = loaded_model_linel.predict(X_new)
        return render_template('lab2.html', title="Линейная регрессия", menu=menu,
                               class_model=pred[0][0])


@app.route("/p_lab3", methods=['POST', 'GET'])
def f_lab3():
    dct = {
        0: "blue",
        2: "green",
        1: "white"
    }
    if request.method == 'GET':
        return render_template('lab3.html', title="Логистическая регрессия", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = loaded_model_logic.predict(X_new)
        return render_template('lab3.html', title="Логистическая регрессия", menu=menu,
                               class_model="Любимый цвет: " + dct[pred[0]])


@app.route("/p_lab4", methods=['POST', 'GET'])
def f_lab4():
    if request.method == 'GET':
        return render_template('lab4.html', title="Дерево решений", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = loaded_model_tree.predict(X_new)
        return render_template('lab4.html', title="Дерево решений", menu=menu,
                               class_model="Любимый цвет: " + pred[0])


@app.route("/p_lab5", methods=['POST', 'GET'])
def p_lab5():
    if request.method == 'GET':
        return render_template('labneiron.html', title="Первый нейрон", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3'])]])
        predictions = model_class.predict(X_new)
        print(predictions)
        return render_template('labneiron.html', title="Первый нейрон", menu=menu,
                               class_model="Это: " + {0: "setosa",
                                                      1: "virginica",
                                                      2: "versicolor"}[np.argmax(predictions)])


@app.route('/p_lab6', methods=['POST', 'GET'])
def p_lab6():
    if request.method == 'GET':
        return render_template('weatherneuro.html', title="Первый нейрон", menu=menu, class_model='')
    if request.method == 'POST':

        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        img_path = os.path.join('static', file.filename)
        file.save(img_path)

        img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
        img = image.img_to_array(img)
        img = img.reshape(1, 784)
        img = img.astype('float32') / 255

        prediction = model_weather.predict(img)
        predicted_class = np.argmax(prediction)
        return render_template('weatherneuro.html', title="Первый нейрон", menu=menu,
                               class_model=predicted_class)


@app.route('/api', methods=['get'])
def get_sort():
    X_new = np.array([[float(request.args.get('list1')),
                       float(request.args.get('list2')),
                       float(request.args.get('list3')),
                       float(request.args.get('list4'))]])
    pred = loaded_model_tree.predict(X_new)

    return jsonify(color=pred[0])


@app.route('/api_v2', methods=['get'])
def get_sort_v2():
    request_data = request.get_json()
    X_new = np.array([[float(request_data['list1']),
                       float(request_data['list2']),
                       float(request_data['list3']),
                       float(request_data['list4'])]])
    pred = loaded_model_tree.predict(X_new)

    return jsonify(color=pred[0])


@app.route('/api_class', methods=['get'])
def predict_classification():
    # Получение данных из запроса http://localhost:5000/api_class?length=5&height=5&width=5
    input_data = np.array([[float(request.args.get('length')),
                            float(request.args.get('height')),
                            float(request.args.get('width'))]])
    print(input_data)
    # input_data = np.array(input_data.reshape(-1, 1))

    # Предсказание
    predictions = model_class.predict(input_data)
    print(predictions)
    result = {0: "setosa", 1: "virginica", 2: "versicolor"}[np.argmax(predictions)]
    print(result)
    # меняем кодировку
    app.config['JSON_AS_ASCII'] = False
    return jsonify(ov=str(result))


if __name__ == "__main__":
    app.run(debug=True)
