from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
CORS(app) 


model = LinearRegression()
X = np.array([[1, 40, 3], [2, 60, 5], [3, 80, 7], [4, 100, 9]])  # distance, speed, traffic
y = np.array([80, 70, 60, 40])  
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    distance = data.get('distance', 0)
    speed = data.get('speed', 0)
    traffic = data.get('traffic', 0)

    prediction = model.predict([[distance, speed, traffic]])
    eco_score = round(float(prediction[0]), 2)

    return jsonify({'eco_score': eco_score})

if __name__ == '__main__':
    app.run(debug=True)
