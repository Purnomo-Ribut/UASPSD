#Modul Library
import streamlit as st
import numpy as np
import pandas as pd

#Modul library Metode 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# #modul library data testing dan training
from sklearn.model_selection import train_test_split

# #modul library score tingkat akurasi
from sklearn.metrics import accuracy_score

def load_dataset():
	url = 'https://raw.githubusercontent.com/Purnomo-Ribut/projek-uas/main/online_classroom_data.csv'
	df = pd.read_csv(url,  header='infer', index_col=False)
	df = df.replace(",",".",regex=True)
	df = df.drop(columns=["Unnamed: 0"])
	return df

st.title('Prediksi Saham PT Bank Rakyat Indonesia (Persero)')
st.write 
("""
    Kelompok : 
    * Purnomo Ribut () 200411100156)
    * 
""")
dataset, modelling, implementasi = st.tabs(["Dataset", "Modelling", "Implementasi"])
