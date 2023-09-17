from flask import Flask, render_template, request
import pickle

app = Flask(__name__,template_folder='template')

# Load the model
try:
    with open('savedmodel.sav', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise Exception("Model file not found. Make sure 'savedmodel.sav' exists.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])
            result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
            return render_template('index.html', result=result)
        except (ValueError, KeyError):
            return render_template('error.html', message="Invalid input data. Please check your input.")
    else:
        return render_template('error.html', message="GET request not supported for prediction. Use POST.")

if __name__ == '__main__':
    app.run(debug=True)