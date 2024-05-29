from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('rainfall_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temperature = float(request.form['temperature'])
    dp_avg = float(request.form['dp_avg'])
    humidity = float(request.form['humidity'])
    slp = float(request.form['slp'])
    visibility = float(request.form['visibility'])
    wind = float(request.form['wind'])
    rainfall = float(request.form['rainfall'])

    # Prepare the input data in the correct format
    input_data = np.array([[temperature, dp_avg, humidity, slp, visibility, wind, rainfall]])
    
    # Predict using the loaded model
    prediction = model.predict(input_data)[0]

    return f'The predicted rainfall is {prediction:.2f} mm'

if __name__ == '__main__':
    app.run(debug=True)
