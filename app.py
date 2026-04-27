from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load only model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data (float use karna better hai)
        features = [float(x) for x in request.form.values()]
        
        # Convert to numpy array
        final_features = np.array(features).reshape(1, -1)
        
        # Prediction
        prediction = model.predict(final_features)[0]
        
        # Output
        if prediction == 1:
            output = "Good Mental Health 😊"
        else:
            output = "Needs Attention ⚠️"
        
        return render_template('index.html', prediction_text=f'Result: {output}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)