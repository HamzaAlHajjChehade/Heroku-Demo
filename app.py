import pickle
import numpy as np
from flask import Flask,render_template,request

app=Flask(__name__)
model=pickle.load(open('USA_Housing.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict/', methods=["POST"])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final_features=[np.array(int_features)]    
    prediction=model.predict(final_features)
    
    output=round(prediction[0],4)
    
    return render_template('index.html',prediction_text='The Price of The House is $ {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True,port=5000)