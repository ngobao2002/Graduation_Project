from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np

# Load the model
model = load_model('my_model_super_update.h5')

# Create Flask app
app = Flask(__name__)

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json(force=True)
    
    # Preprocess data if needed
    # Assuming input is a list of values, modify this according to your input data
    input_data = np.array(data['input_data'])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Assuming prediction is a single value, modify this according to your output
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
