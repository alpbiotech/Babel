from flask import Flask, request, jsonify
from flask_cors import CORS
from modules.main import main

app = Flask(__name__)
CORS(app, origins="*")


@app.route('/')
def hello_world():
    return 'Hello Alp AI!'


@app.route('/echo', methods=['POST'])  # test echo POST call
def echo():
    data = request.json
    input_sequence = data.get('inputSequence', '')
    return jsonify({
        'outputSequence': f"OUT_{input_sequence}"
    })


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the POST request
    data = request.json
    input_sequence = data.get('inputSequence', '')
    predictions: dict = main(input_sequence)
    return jsonify(predictions)


if __name__ == '__main__':
    app.run()
