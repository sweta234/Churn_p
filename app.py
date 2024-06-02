from sklearn.impute  import SimpleImputer
#from feature_engine.encoding import RareLabelEncoder
import warnings
from sklearn.preprocessing import (
    OneHotEncoder)
import sklearn 
from sklearn.pipeline import Pipeline
from feature_engine.encoding import RareLabelEncoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle

import joblib
import streamlit as st
import warnings
import altair as alt


from sklearn.compose import ColumnTransformer
# Define the columns to be preprocessed
categorical_cols = ['Surname', 'Gender', 'Geography']

sklearn.set_config(transform_output="pandas")

# Define the preprocessor pipeline
preprocessor = Pipeline(
    steps=[
        ('rare_label', RareLabelEncoder(tol=0.05, n_categories=1, variables=categorical_cols)),
        ('onehot', ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'  # This keeps other columns unchanged
        ))
    ]
)

path = r"C:\Users\win10\OneDrive\Desktop\CHURN_PREDICTIONS\Data\train.csv"
train = pd.read_csv(path)
X_train = train.drop(columns='Exited')
y_train = train['Exited']

preprocessor.fit(X_train, y_train)

joblib.dump(preprocessor, "preprocessor.joblib")


st.set_page_config(
	page_title="Churn Prediction",
	layout="wide"
)

st.title("Churn Prediction - AWS SageMaker")

Geography = st.selectbox(
	"Geography:",
	options=X_train.Geography.unique()
)

Gender = st.selectbox(
	"Gender:",
	options=X_train.Gender.unique()
)

Credit_Score = st.number_input(
	"Credit_Score:",
	step=1
)




Age = st.number_input(
	"Age:",
	step=1
)

Tenure = st.number_input(
	"Tenure:",
	step=1
)
Balance= st.number_input(
	"Balance:",
	step=1
)
Number_of_product = st.number_input(
	"Number_of_product:",
	step=1
)
Has_credit = st.number_input(
	"Has_credit:",
	step=1
)

IsActiveMember = st.number_input(
	"IsActiveMember:",
	step=1
)
EstimatedSalary = st.number_input(
	"EstimatedSalary:",
	step=1
)


x_new = pd.DataFrame(dict(
	Geography=[Geography],
	Gender=[Gender],
	Credit_Score=[Credit_Score],
	Age =[Age],
	Tenure=[Tenure],
	Balance=[Balance],
	Number_of_product=[Number_of_product],
	Has_credit=[Has_credit],
	IsActiveMember=[IsActiveMember],
    EstimatedSalary=[EstimatedSalary]
))


if st.button("Predict"):
	saved_preprocessor = joblib.load("preprocessor.joblib")
	x_new_pre = saved_preprocessor.transform(x_new)

	with open("xgboost-model", "rb") as f:
		model = pickle.load(f)
	x_new_xgb = xgb.DMatrix(x_new_pre)
	pred = model.predict(x_new_xgb)[0]

	st.info(f"The predicted price is {pred:,.0f} INR")