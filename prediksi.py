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
    df = pd.read_csv("BBNI.JK.csv")
    df

    def split_sequence(sequence, n_steps):
    # Inisialisasi list kosong untuk menyimpan input (X) dan output (y)
        X, y = list(), list()
        for i in range(len(sequence)):
            # Menemukan akhir pola ini berdasarkan jumlah langkah (n_steps)
            end_ix = i + n_steps
            # Memeriksa apakah kita sudah melebihi urutan
            if end_ix > len(sequence)-1:
                break
            # Mengumpulkan bagian input dan output dari pola
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)

    # Mengembalikan X dan y dalam bentuk array NumPy
        return array(X), array(y)
    n = len(df)
    # Menghitung jumlah total data dalam dataset
    # dan menyimpannya dalam variabel n


    sizeTrain = (round(n*0.8))
    # Menghitung jumlah data yang akan digunakan untuk data latih
    # dengan mengalikan 0.8 (80%) dengan total jumlah data
    # dan membulatkannya ke bilangan terdekat menggunakan fungsi round()
    # dan menyimpan hasilnya dalam variabel sizeTrain

    data_train = pd.DataFrame(df[:sizeTrain])
    train_array = data_train.values  # Mengonversi DataFrame menjadi numpy array
    train_scaled = scaler.fit_transform(train_array)  # Melakukan normalisasi pada numpy array
    # Membuat DataFrame baru untuk data latih
    # dengan menggunakan slicing untuk mengambil data dari indeks 0 sampai sizeTrain
    # dan menyimpannya dalam variabel data_train

    data_test = pd.DataFrame(df[sizeTrain:])
    # Membuat DataFrame baru untuk data uji
    # dengan menggunakan slicing untuk mengambil data mulai dari indeks sizeTrain hingga akhir
    # dan menyimpannya dalam variabel data_test

    #dates_test = pd.DataFrame(dates[sizeTrain:])
    # Komentar ini memberikan penjelasan bahwa ada variabel dates yang tidak digunakan di sini

    data_train
    # Menampilkan DataFrame data_train


    st.write("Normalisasi data menggunakan MinMaxScaler")
    # Normalisasi data menggunakan MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    # Mengimport MinMaxScaler dari library sklearn.preprocessing
    # dan membuat objek scaler dari kelas tersebu

    train_scaled = scaler.fit_transform(data_train)
    # Menggunakan scaler.fit_transform untuk melakukan normalisasi data pada data latih (data_train)
    # Normalisasi dilakukan agar nilai-nilai data berada dalam rentang [0, 1]
    # Menghasilkan array yang berisi data latih yang telah dinormalisasi dan disimpan dalam variabel train_scaled

    test_scaled = scaler.transform(data_test)
    # Menggunakan scaler.transform untuk menerapkan normalisasi yang sama pada data uji (data_test)
    # Menggunakan transform() karena kita ingin menggunakan parameter yang telah dihitung
    # pada proses normalisasi data latih (data_train)
    # Menghasilkan array yang berisi data uji yang telah dinormalisasi dan disimpan dalam variabel test_scaled

    # reshaped_data = data.reshape(-1, 1)
    # Komentar ini memberikan penjelasan bahwa ada variabel reshaped_data yang tidak digunakan di sini

    train = pd.DataFrame(train_scaled, columns = ['df'])
    # Membuat DataFrame baru untuk data latih yang telah dinormalisasi (train_scaled)
    # dengan kolom bernama 'data'

    train = train['df']
    # Mengambil kolom 'data' dari DataFrame train dan menyimpannya kembali dalam variabel train

    test = pd.DataFrame(test_scaled, columns = ['df'])
    # Membuat DataFrame baru untuk data uji yang telah dinormalisasi (test_scaled)
    # dengan kolom bernama 'data'

    test = test['df']
    # Mengambil kolom 'data' dari DataFrame test dan menyimpannya kembali dalam variabel test

    test
    # Menampilkan DataFrame test


with modelling :
    st.write("Prepocessing dimulai dari : ") 
with implementasi :
    st.write("Prepocessing dimulai dari : ") 
        