from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load your pre-trained modell
model = joblib.load('app/flight.pkl')


@app.route('/')
def home():
    return render_template('Flightdelay.html')


@app.route('/result', methods=['POST'])
def predict():
    fl_num = int(request.form.get('fno'))
    month = int(request.form.get('month'))
    dayofmonth = int(request.form.get('daym'))
    dayofweek = int(request.form.get('dayw'))
    sdeptime = int(request.form.get('sdt'))
    adeptime = int(request.form.get('adt'))
    arrtime = int(request.form.get('sat'))
    depdelay = int(adeptime) - int(sdeptime)

    inputs = list()
    inputs.append(fl_num)
    inputs.append(month)
    inputs.append(dayofmonth)
    inputs.append(dayofweek)

    # Set delay flag based on departure delay
    if depdelay < 15:
        inputs.append(0)  # Flight On Time
    else:
        inputs.append(1)  # Flight Delayed

    inputs.append(arrtime)

    origin = str(request.form.get("org"))
    dest = str(request.form.get("dest"))

    # One-hot encoding for origin airports (ensure origin is valid)
    if origin == 'ATL':
        a = [1, 0, 0, 0, 0]
    elif origin == 'DTW':
        a = [0, 1, 0, 0, 0]
    elif origin == 'JFK':
        a = [0, 0, 1, 0, 0]
    elif origin == 'MSP':
        a = [0, 0, 0, 1, 0]
    elif origin == 'SEA':
        a = [0, 0, 0, 0, 1]
    else:
        # Default to 'ATL' if no valid origin is provided
        a = [1, 0, 0, 0, 0]

    inputs.extend(a)

    # One-hot encoding for destination airports (ensure destination is valid)
    if dest == 'ATL':
        b = [1, 0, 0, 0, 0]
    elif dest == 'DTW':
        b = [0, 1, 0, 0, 0]
    elif dest == 'JFK':
        b = [0, 0, 1, 0, 0]
    elif dest == 'MSP':
        b = [0, 0, 0, 1, 0]
    elif dest == 'SEA':
        b = [0, 0, 0, 0, 1]
    else:
        # Default to 'ATL' if no valid destination is provided
        b = [1, 0, 0, 0, 0]

    inputs.extend(b)

    # Make prediction
    prediction = preprocessAndPredict(inputs)

    # Convert prediction (0 or 1) to message
    if prediction == 1:
        result_message = "Flight Delayed"
    else:
        result_message = "Flight On Time"

    return render_template('result.html', result_message=result_message)


def preprocessAndPredict(inputs):
    # Ensure that inputs have the correct number of features
    try:
        # Reshape the input data to match the model's expected input
        test_data = np.array(inputs).reshape((1, -1))
        prediction = model.predict(test_data)
        return prediction[0]  # Return the first (and only) prediction result
    except Exception as e:
        print(f"Error in preprocessing and prediction: {str(e)}")
        return None


if __name__ == '__main__':
    app.run(debug=True)
