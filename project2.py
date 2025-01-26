from flask import Flask, render_template, jsonify, request
import numpy as np
import pickle

app=Flask(__name__)

lasso=pickle.load(open('lassocv.pkl','rb'))
scale=pickle.load(open('scale.pkl','rb'))

@app.route('/')
def welcome():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Class=int(request.form.get('Classes'))
        Region=int(request.form.get('Region'))
        x=np.array([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Class,Region]])
        x=scale.transform(x)
        pred=lasso.predict(x)[0]
        return render_template('data.html',result='%.3f'%pred)
    else:
        return render_template('data.html')

if(__name__=='__main__'):
    app.run(host='0.0.0.0')