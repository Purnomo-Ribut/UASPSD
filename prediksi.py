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
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        destree = st.checkbox('Decission Tree')
        mlp = st.checkbox('Multi-Layer Perceptron')
        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)
    
        y_compare = np.vstack((test_label,y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        # MLP
        mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
        mlp.fit(training, training_label)
        mlp_predict = mlp.predict(test)

        mlp_akurasi = round(100 * accuracy_score(test_label, mlp_predict))

        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
            if mlp :
                st.write("Model MLP accuracy score : {0:0.2f}" . format(mlp_akurasi))
with implementasi : 
   with st.form("my_form"):
        st.subheader("Implementasi")
        Open = st.number_input('Masukkan Harga pembukaan : ')
        High = st.number_input('Masukkan Harga tertinggi : ')
        Low = st.number_input('Masukkan Harga terendah : ')
        Close = st.number_input('Masukkan Harga penutupan : ')
        Adj_Close = st.number_input('Masukkan Harga penutupan saham yang telah disesuaikan : ')
        Volume = st.number_input('Masukkan Jumlah saham yang diperdagangkan : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes', 'Decision Tree', 'Multi-Layer Perceptron'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                Open,
                High,
                Low,
                Close,
                Adj_Close,
                Volume,
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian 
            if model == 'Decision Tree':
                mod = dt
            if model == 'Multi-Layer Perceptron':
                mod = mlp

            input_pred = mod.predict(input_norm)


            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)
