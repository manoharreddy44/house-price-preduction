import pandas as pd
from sklearn.linear_model import LinearRegression
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app)

# Load the dataset
data = pd.read_csv('Housing.csv')


if 'city' not in data.columns:

    data['city'] = 'UNKNOWN'


encoder = LabelEncoder()

binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']


for col in binary_columns:
    data[col] = data[col].map({'yes': 1, 'no': 0})


data['furnishingstatus'] = encoder.fit_transform(data['furnishingstatus'])


data = pd.get_dummies(data, columns=['city'], drop_first=True)


X = data.drop('price', axis=1)
y = data['price']

model = LinearRegression()

# Train the model
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.json  

   
        for col in binary_columns:
            user_input[col] = 1 if user_input[col] == 'yes' else 0


        if user_input['furnishingstatus'] in encoder.classes_:
            user_input['furnishingstatus'] = encoder.transform([user_input['furnishingstatus']])[0]
        else:
            return jsonify({"error": "Invalid input for furnishingstatus. Please enter 'furnished', 'semi-furnished', or 'unfurnished'."})

        # Add a new 'city' column based on user input
        city_name = user_input.pop('city')
        for col in X.columns:
            if col.startswith('city_'):
                user_input[col] = 1 if city_name.lower() == col[5:].lower() else 0

        # Convert user input to a DataFrame
        user_df = pd.DataFrame([user_input])

        # Make predictions
        predicted_price_inr = model.predict(user_df)[0]  

        return jsonify({"predicted_price_inr": predicted_price_inr})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
