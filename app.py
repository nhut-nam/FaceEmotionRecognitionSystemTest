from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin
import os
from FacialExpressionRecognition.utils.common import decode_image
from FacialExpressionRecognition.pipeline.prediction import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self, app):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(filename=self.filename)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def train():
    os.system("dvc repro")
    return "Training done successfully!!"

@app.route("/predict", methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        message = request.json['image']
        decode_image(message, clApp.filename)
        prediction, accuracy = clApp.classifier.predict()
        print(prediction)
        return jsonify({"prediction": prediction, "accuracy": accuracy})
    else:
        return render_template('templates/index.html')


if __name__ == "__main__":
    clApp = ClientApp(app)
    app.run(host='0.0.0.0', port=8080)