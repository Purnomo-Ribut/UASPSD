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
	url = 'https://github.com/Purnomo-Ribut/UASPSD/blob/72ab2198224d48f894e43f83f862b3e55fab4856/BBNI.JK.csv'
	df = pd.read_csv(url,  header='infer', index_col=False)
	df = df.replace(",",".",regex=True)
	df = df.drop(columns=["Unnamed: 0"])
	return df

st.title('Prediksi Saham PT Bank Rakyat Indonesia (Persero)')
st.write ("""
    Kelompok : 
    * Purnomo Ribut (200411100156)
    * Dhafa Febriyan Wiranata (200411100169)
""")
data, Prepocessing, modelling, implementasi = st.tabs(["Dataset","Prepocessing Data" ,"Modelling", "Implementasi"])

with data:
    st.write("Data ini didapatkan dari")

with Prepocessing : 
    st.write("Prepocessing dimulai dari : ")
with modelling : 
    st.write("Modelling terdapat 3 : ")
with implementasi : 
    st.title("Prediksi Saham Terkini dengan memasukkkan input ft ")
