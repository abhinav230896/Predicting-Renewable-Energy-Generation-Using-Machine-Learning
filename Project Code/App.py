import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__,template_folder='template')
model_wind = pickle.load(open('model_wind.pkl', 'rb'))
model_slr = pickle.load(open('model_slr.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_solar',methods=['POST'])
def predict_solar():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model_slr.predict(final_features)
    output=round(prediction[0],2)
    return render_template('index.html', prediction_solar='Solar Generation : {} MW'.format(output))

@app.route('/predict_wind',methods=['POST'])
def predict_wind():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model_wind.predict(final_features)
    output=round(prediction[0],2)
    return render_template('index.html', prediction_wind='Wind Generation : {} MW'.format(output))


if __name__ == "__main__":
    app.run(debug=True)

# used to execute some code only if the file was run directly, and not imported.