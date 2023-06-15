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
	data = pd.read_csv("BBNI.JK.csv")		
	return data

st.title('Prediksi Saham PT Bank Rakyat Indonesia (Persero)')
st.write ("""
    Kelompok : 
    * Purnomo Ribut (200411100156)
    * Dhafa Febriyan Wiranata (200411100169)
""")
data, Prepocessing, modelling, implementasi = st.tabs(["Dataset","Prepocessing Data" ,"Modelling", "Implementasi"])

with data:
    st.write("Tampilan Dataset Saham PT Bank Rakyat Indonesia Persero")
    st.dataframe(load_dataset())
    st.write("""
    Dataset Saham PT Bank Rakyat Indonesia Persero yang diambil dari finance.yahoo.com berisi informasi harga saham pada setiap tanggal perdagangan. 
    Berikut adalah deskripsi dari setiap kolom dalam dataset 

    * Date: Tanggal perdagangan saham.
    * Open: Harga pembukaan (opening price) saham pada hari perdagangan.
    * High: Harga tertinggi (highest price) yang dicapai oleh saham pada hari perdagangan.
    * Low: Harga terendah (lowest price) yang dicapai oleh saham pada hari perdagangan.
    * Close: Harga penutupan (closing price) saham pada hari perdagangan.
    * Adj Close: Harga penutupan saham yang telah disesuaikan (adjusted closing price) dengan faktor-faktor seperti pembagian saham, dividen, dan aksi korporasi lainnya.
    * Volume: Jumlah saham yang diperdagangkan pada hari perdagangan.


    Dataset ini dapat digunakan untuk menganalisis pergerakan harga saham PT Bank Rakyat Indonesia serta volume perdagangan yang terjadi dalam rentang waktu tertentu. Dengan menggunakan informasi ini, dapat dilakukan berbagai analisis seperti peramalan harga saham, identifikasi tren pasar, dan pengambilan keputusan investasi.""")



with Prepocessing : 
    st.write("Prepocessing dimulai dari : ")
    

with modelling :
    #X_norm, x_test, training_y, y_test = train_test_split(X_norm, test_size=0.2, random_state=1)
    x_test = test.iloc[:, :6]
    y_test = test.iloc[:, 6:]
    y_test.set_axis(["y_test"], axis="columns")

    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighbors')
        destree = st.checkbox('Decision Tree')
        submitted = st.form_submit_button("Submit")

        if naive:
            model_n = GaussianNB()
            model_n.fit(X_norm, training_y)
            y_pred3=model_n.predict(x_test)
            #gaussian_accuracy = round(100 * accuracy_score(y_test, y_pred3), 2)
            from sklearn.metrics import mean_absolute_percentage_error
            mape = mean_absolute_percentage_error(y_test, y_pred3)
            #st.write('Model Gaussian Naive Bayes accuracy score:', gaussian_accuracy)
            st.write('MAPE Model Gaussian Naive Bayes:', mape)

        if k_nn:
            # import knn
            from sklearn.neighbors import KNeighborsRegressor
            model_knn = KNeighborsRegressor(n_neighbors=30)
            model_knn.fit(X_norm, training_y)
            y_pred2=model_knn.predict(x_test)
            # knn_accuracy = round(100 * accuracy_score(y_test, y_pred2), 2)    
            from sklearn.metrics import mean_absolute_percentage_error
            mape = mean_absolute_percentage_error(y_test, y_pred2)
            # st.write("Model K-Nearest Neighbors accuracy score:", knn_accuracy )
            st.write('MAPE Model K-Nearest Neighbors :', mape)

        if destree:
            #klasifikasi menggunakan decision tree
            model_tree = tree.DecisionTreeClassifier(random_state=3, max_depth=1)
            model_tree.fit(X_norm, training_y)
            y_pred1=model_tree.predict(x_test)
            #dt_accuracy = round(100 * accuracy_score(y_test, y_pred1), 2)
            from sklearn.metrics import mean_absolute_percentage_error 
            mape = mean_absolute_percentage_error(y_test, y_pred1) 
            #st.write("Model Decision Tree accuracy score:", dt_accuracy)
            st.write('MAPE Model Decision Tree:', mape)

        
with implementasi :
    st.write("Prepocessing dimulai dari : ") 
        