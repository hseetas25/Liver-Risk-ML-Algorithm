#flask library
from flask import Flask, render_template, request

#pickle library
import pickle

#pandas and numpy (dataframe) library
import pandas as pd
import numpy as np
#from sklearn.neighbors import _dist_metrics

#CORS Policy library
from flask_cors import CORS, cross_origin

#Flask main application
app = Flask(__name__,template_folder='view')

#open and read the RF-model
RFModel = pickle.load(open('smote_RandomForest.pickle','rb'))
adbModel = pickle.load(open('smote_adb.pickle','rb'))
lrModel = pickle.load(open('smote_Logistic.pickle','rb'))
#dtreeModel = pickle.load(open('DecisionTree.pickle','rb'))
#KNNModel = pickle.load(open('Knn.pickle','rb'))

@cross_origin()
@app.route("/", methods = ['GET', 'POST'])
def home():
    return render_template('index.html')

@cross_origin()
@app.route('/predict-liver-tumor',methods=['POST'])
def predictTumor():
    age = int(request.form.get('age'))
    totalbilirubin = float(request.form.get('totalbilirubin'))
    directbilirubin = float(request.form.get('directbilirubin'))
    alkaline = float(request.form.get('alkaline'))
    alamine = float(request.form.get('alamine'))
    aspartate = float(request.form.get('aspartate'))
    protiens = float(request.form.get('protiens'))
    albumin = float(request.form.get('albumin'))
    ratio  = float(request.form.get('ratio'))
    predictionOfRFModel = RFModel.predict(pd.DataFrame(columns=['Age', 'Total_Bilirubin','Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio'],
                 data = np.array([age,totalbilirubin,directbilirubin, alkaline, alamine,aspartate,protiens, albumin, ratio]).reshape(1,9)))
    predictionOfadbModel = adbModel.predict(pd.DataFrame(columns=['Age', 'Total_Bilirubin','Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio'],
                 data = np.array([age,totalbilirubin,directbilirubin, alkaline, alamine,aspartate,protiens, albumin, ratio]).reshape(1,9)))
    predictionOflrModel = lrModel.predict(pd.DataFrame(columns=['Age', 'Total_Bilirubin','Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio'],
                 data = np.array([age,totalbilirubin,directbilirubin, alkaline, alamine,aspartate,protiens, albumin, ratio]).reshape(1,9)))
    #predictionOfdtreeModel = dtreeModel.predict(pd.DataFrame(columns=['Age', 'Total_Bilirubin','Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio'],
                 #data = np.array([age,totalbilirubin,directbilirubin, alkaline, alamine,aspartate,protiens, albumin, ratio]).reshape(1,9)))
    #predictionOfKNNModel = KNNModel.predict(pd.DataFrame(columns=['Age', 'Total_Bilirubin','Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio'],
                 #data = np.array([age,totalbilirubin,directbilirubin, alkaline, alamine,aspartate,protiens, albumin, ratio]).reshape(1,9)))
    #prediction
    #print(predictionOfRFModel[0],predictionOfadbModel[0],predictionOflrModel[0])
    l=[predictionOfRFModel[0],predictionOfadbModel[0],predictionOflrModel[0]]
    if l.count(1)>=l.count(2):
        return str(1)
    return str(2)

if(__name__=="__main__"):
    app.run(debug=True)