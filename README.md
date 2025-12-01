# 🏦 Bank Campaign Term Deposit Prediction

## 1. Business Understanding  
### 1.1 Latar Belakang
Industri perbankan menghadapi tantangan signifikan dalam mengoptimalkan kampanye pemasaran langsung, khususnya melalui telemarketing, dimana seringkali sulit bagi pemasar bank untuk secara efektif menjual layanan atau produk keuangan dalam waktu singkat melalui komunikasi telepon[[1]](https://doi.org/10.1016/j.dss.2014.03.001). Krisis keuangan global tahun 2008 mengubah lanskap perbankan secara drastis di Eropa, di mana bank-bank Portugal mengalami tekanan untuk meningkatkan persyaratan modal melalui penggalangan deposito jangka panjang yang lebih besar [[2]](https://doi.org/10.1016/j.dss.2014.03.001). Telemarketing sebagai salah satu strategi direct marketing menghadapi permasalahan kompleks mulai dari biaya operasional tinggi, tingkat keberhasilan yang rendah hanya mencapai 12,38% dalam dataset penelitian ini hingga potensi gangguan terhadap nasabah dan kesulitan dalam menyeleksi target yang tepat. Pemanfaatan pendekatan data mining dan machine learning menjadi solusi potensial untuk membangun Decision Support System (DSS) yang mampu memprediksi kesuksesan kampanye telemarketing sebelum panggilan dilakukan, sehingga dapat meningkatkan efisiensi dan efektivitas kampanye pemasaran bank.

### 1.2 Rumusan Masalah
1. Bagaimana memprediksi keberhasilan kampanye telemarketing bank untuk produk deposito jangka panjang?
2. Fitur-fitur apa saja yang paling berpengaruh terhadap keberhasilan kampanye telemarketing?
3. Model machine learning mana yang memberikan performa terbaik untuk prediksi ini?
4. Bagaimana implementasi model terbaik dalam bentuk web application yang dapat digunakan oleh manajer kampanye?

### 1.3 Tujuan 
1. Membangun model prediktif yang akurat untuk memprediksi keberhasilan telemarketing
2. Mengidentifikasi faktor-faktor kunci yang mempengaruhi keputusan nasabah
3. Membandingkan performa berbagai algoritma machine learning (Random Forest, Support Vector Machine, K-Nearest Neighbors, Gradient Boosting)
4. Mengembangkan web application di Platform Hugging Face untuk deployment model

### 1.4 Manfaat
1. Meningkatkan efisiensi kampanye dengan mengurangi jumlah panggilan yang tidak produktif
2. Menghemat biaya operasional telemarketingMemberikan kontribusi dalam pengembangan metode feature engineering
3. Menambah literatur tentang penerapan machine learning dalam sektor perbankan
---

## 2. Data Understanding  

### 2.1 Deskripsi dataset

**Nama:** [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)  
**Sumber:** UCI Machine Learning Repository
**Jumlah:** 45.211 sampel dengan 17 kolom

#### A. Informasi Nasabah
| Fitur     | Tipe        | Deskripsi                                                        |
|-----------|-------------|------------------------------------------------------------------|
| age       | Numerik     | Usia nasabah                                                     |
| job       | Kategorikal | Jenis pekerjaan (admin, blue-collar, entrepreneur, dll)          |
| marital   | Kategorikal | Status pernikahan (married, single, divorced)                    |
| education | Kategorikal | Tingkat pendidikan (primary, secondary, tertiary, unknown)       |
| default   | Kategorikal | Apakah memiliki kredit macet (yes/no)                            |
| balance   | Numerik     | Saldo rata-rata tahunan (dalam euro)                             |
| housing   | Kategorikal | Apakah memiliki pinjaman perumahan (yes/no)                      |
| loan      | Kategorikal | Apakah memiliki pinjaman pribadi (yes/no)                        |

#### B. Informasi Kontak Kampanye
| Fitur    | Tipe        | Deskripsi                                           |
|----------|-------------|-----------------------------------------------------|
| contact  | Kategorikal | Tipe komunikasi (cellular, telephone, unknown)      |
| day      | Numerik     | Hari dalam bulan ketika nasabah dihubungi           |
| month    | Kategorikal | Bulan terakhir kontak dalam setahun                 |
| duration | Numerik     | Durasi kontak terakhir (dalam detik)                |

#### C. Informasi Kampanye Sebelumnya
| Fitur    | Tipe        | Deskripsi                                                         |
|----------|-------------|-------------------------------------------------------------------|
| campaign | Numerik     | Jumlah kontak dalam kampanye ini                                  |
| pdays    | Numerik     | Hari sejak terakhir dihubungi dari kampanye sebelumnya            |
| previous | Numerik     | Jumlah kontak sebelum kampanye ini                                |
| poutcome | Kategorikal | Hasil kampanye sebelumnya (success, failure, other, unknown)      |

#### D. Target Variable
| Fitur | Tipe        | Deskripsi                                       |
|-------|-------------|--------------------------------------------------|
| y     | Kategorikal | Apakah nasabah berlangganan deposito (yes/no)    |


### 2.2 Eksplorasi Data

![Korelasi Antar Variabel](https://github.com/adstika20/uts_datascience/blob/main/Image/heatmap%20korelasi.png)

Sebagian besar fitur numerik memiliki korelasi yang sangat lemah (|r| < 0.2), mengindikasikan independensi informasi antar fitur dan tidak adanya masalah multicollinearity serius. Korelasi tertinggi terjadi antara `pdays` dan `previous` (r = 0.45), yang logis karena keduanya merepresentasikan informasi kampanye sebelumnya dan menunjukkan overlap informasi yang perlu diperhatikan dalam feature engineering. Fitur `day` berkorelasi lemah dengan `campaign` (r = 0.16), mengindikasikan strategi push di akhir bulan dengan lebih banyak attempt kontak. `Duration` menunjukkan korelasi negatif sangat lemah dengan campaign (r = -0.08), menandakan durasi percakapan cenderung lebih pendek saat nasabah telah dihubungi berkali-kali. 

## 3. Data Preparation  

### 3.1 Label Encoder 
Label Encoding diterapkan pada 10 fitur kategorikal dalam dataset menggunakan LabelEncoder dari scikit-learn. Fitur-fitur yang di-encode meliputi variabel target y (yes=1, no=0) serta fitur-fitur seperti job, marital, education, default, housing, loan, contact, month, dan poutcome. 

### 3.2 Pembagian Data
Dataset dibagi dengan proporsi 80:20, dimana 80% data digunakan untuk melatih model (training set) dan 20% sisanya digunakan untuk menguji performa model (testing set). Dengan total **45.211** sampel dalam dataset, pembagian ini menghasilkan sekitar **36.168** sampel untuk training dan **9.043** sampel untuk testing, jumlah yang cukup substansial untuk keduanya.

### 3.2 Data Standardization
Standarisasi merupakan teknik preprocessing yang mentransformasi fitur-fitur numerik sehingga memiliki distribusi dengan mean (rata-rata) 0 dan standard deviation (simpangan baku) 1. 

**Metode:** StandardScaler (Z-score normalization)

**Formula:**
```
z = (x - μ) / σ
```
Dimana:
- μ = mean
- σ = standard deviation
  
Scaler di-fit hanya pada training data (fit_transform) untuk menghitung mean dan standard deviation, kemudian transformasi yang sama diterapkan pada test data (transform) menggunakan statistik dari training set. Pendekatan ini mencegah data leakage dari test set ke training set, menjaga integritas evaluasi model.

---

## 4. Modeling  

#### **4.1 Random Forest (RF)**

RF adalah algoritma ensemble learning yang mengkombinasikan multiple decision trees untuk menghasilkan prediksi yang lebih robust dan akurat. Metode ini bekerja dengan membangun sejumlah decision trees pada subset data training yang berbeda (bootstrap sampling) dan melakukan voting mayoritas untuk klasifikasi atau rata-rata untuk regresi[[3]](https://link.springer.com/article/10.1023/A:1010933404324). Model RF yang diimplementasikan menggunakan 100 estimator (pohon) dengan parameter n_jobs=-1 untuk memanfaatkan semua CPU cores yang tersedia. 

#### **2. Support Vector Machine (SVM)**

SVM adalah algoritma supervised learning yang bekerja dengan mencari hyperplane optimal di ruang berdimensi tinggi untuk memisahkan kelas-kelas data dengan margin maksimum[[4]](https://link.springer.com/article/10.1007/BF00994018). Implementasi SVM menggunakan Radial Basis Function (RBF) kernel secara default dengan parameter probability=True untuk menghasilkan probability estimates melalui Platt scaling.  

#### **3. Gradient Boosting Classifier (GBC)**

GBC adalah teknik ensemble yang membangun model secara sekuensial, di mana setiap model baru difokuskan untuk memperbaiki kesalahan prediksi dari model sebelumnya melalui gradient descent optimization [[5]](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full#:~:text=October%202001).%20DOI%3A-,10.1214/aos/1013203451,-ABOUT). Algoritma ini bekerja dengan menambahkan weak learners secara iteratif, di mana setiap learner baru meminimalkan loss function dengan memprediksi residual error dari ensemble sebelumnya. Model GBC yang digunakan memiliki 200 estimator dengan learning rate 0.1 dan max_depth 3 untuk mengontrol kompleksitas setiap tree. 

#### **4. K-Nearst Neighbors (KNN)**
K-Nearest Neighbors adalah algoritma instance-based learning yang melakukan klasifikasi berdasarkan voting mayoritas dari k tetangga terdekat dalam feature space [[6]](https://doi.org/10.1109/TIT.1967.1053964). Algoritma ini tidak membangun model eksplisit selama training, melainkan menyimpan seluruh training instances dan melakukan komputasi saat prediction dengan menghitung jarak antara instance baru dengan semua training instances. Model KNN yang diimplementasikan menggunakan k=5 neighbors dengan Euclidean distance sebagai metrik default. 

---

## 5. Evaluation  

### 5.1 Classification Report
- Accuracy : menghitung berapa banyak prediksi (baik positif maupun negatif) yang sesuai dengan nilai sebenarnya dibandingkan dengan total seluruh prediksi.
- Precision : mengukur tingkat ketepatan prediksi positif yang dibuat oleh model.
- Recall : mengukur kemampuan model dalam menemukan semua instance positif yang sebenarnya ada dalam dataset.
- F1 Score : mean dari precision dan recall yang memberikan keseimbangan antara kedua metrik tersebut.

| Model | Class | Precision | Recall | F1-Score | Accuracy |
|-------|-------|-----------|--------|----------|----------|
| RF | 0 | 0.93 | **0.97** | **0.95** | **0.91** |
| RF | 1 | **0.66** | 0.42 | 0.51 | |
| SVM | 0 | 0.91 | **0.98** | 0.94 | 0.90 |
| SVM | 1 | 0.65 | 0.26 | 0.37 | |
| GBC | 0 | **0.93** | **0.97** | **0.95** | **0.91** |
| GBC | 1 | **0.66** | **0.43** | **0.52** | |
| KNN | 0 | 0.91 | **0.97** | 0.94 | 0.89 |
| KNN | 1 | 0.57 | 0.32 | 0.41 | |

*Catatan: Class 0 = Tidak berlangganan, Class 1 = Berlangganan deposito*

**Interpretasi Hasil**

Berdasarkan tabel di atas, **Random Forest (RF)** dan **Gradient Boosting Classifier (GBC)** menunjukkan performa terbaik dengan accuracy yang sama yaitu **91%**. Kedua model ini juga memiliki F1-Score tertinggi untuk class 0 (kelas mayoritas) sebesar 0.95. Untuk kelas minoritas (class 1 - nasabah yang berlangganan), GBC sedikit unggul dengan recall 0.43 dan F1-Score 0.52, dibandingkan RF yang hanya mencapai recall 0.42 dan F1-Score 0.51. Hal ini menunjukkan bahwa GBC lebih baik dalam mendeteksi nasabah potensial yang akan berlangganan deposito.

**Support Vector Machine (SVM)** menempati posisi kedua dengan accuracy 90%, namun memiliki performa yang kurang memuaskan pada class 1 dengan recall terendah yaitu 0.26. Artinya, SVM hanya mampu mengidentifikasi 26% dari nasabah yang sebenarnya berlangganan deposito, sehingga model ini akan melewatkan banyak peluang potensial. Meskipun SVM memiliki precision yang cukup baik (0.65) untuk class 1, recall yang sangat rendah membuat F1-Score hanya mencapai 0.37, yang merupakan yang terburuk di antara semua model.

**K-Nearest Neighbors (KNN)** menunjukkan performa yang paling rendah dengan accuracy 89%. Untuk class 1, KNN memiliki precision terendah (0.57) yang berarti banyak prediksi positif yang sebenarnya adalah false positive. Recall KNN untuk class 1 juga rendah (0.32), mengindikasikan bahwa model ini melewatkan banyak nasabah potensial. Kombinasi precision dan recall yang rendah menghasilkan F1-Score 0.41, menjadikan KNN sebagai pilihan yang kurang optimal untuk kasus ini.

**Rekomendasi model terbaik adalah Gradient Boosting Classifier (GBC)** karena memiliki keseimbangan terbaik antara precision dan recall untuk class 1 (kelas target yang lebih penting dalam konteks bisnis). Meskipun accuracy RF dan GBC sama-sama 91%, GBC lebih unggul dalam menangkap nasabah potensial (recall class 1 lebih tinggi) dengan tetap mempertahankan precision yang baik. Dalam konteks telemarketing bank, kemampuan untuk mengidentifikasi lebih banyak nasabah potensial (recall tinggi) sambil meminimalkan false positive (precision tinggi) sangat krusial untuk efisiensi kampanye dan ROI yang optimal.


### 5.4 Confusion Matrix 

| RF | SVM |
|---------------------|-----------------------|
| ![RF](https://github.com/adstika20/uts_datascience/blob/main/Image/confussion%20matrix%20RF.png) | ![SVM](https://github.com/adstika20/uts_datascience/blob/main/Image/confussion%20matrix%20SVM.png) |

| GBC | KNN |
|---------------|-----|
| ![GBC](https://github.com/adstika20/uts_datascience/blob/main/Image/confussion%20matrix%20gbc.png) | ![KNN](https://github.com/adstika20/uts_datascience/blob/main/Image/confussion%20matrix%20knn.png) |

Model **GBC** dan **RF** menunjukkan performa identik dengan kemampuan terbaik dalam memprediksi kelas mayoritas (class 0), dimana keduanya berhasil mengklasifikasikan 7752 True Negative dengan benar dan hanya 233 False Positive. Untuk kelas minoritas (class 1 - nasabah berlangganan), GBC dan RF mampu mendeteksi 445 True Positive namun masih melewatkan 613 False Negative, mengindikasikan bahwa sekitar 58% nasabah potensial masih terlewatkan. **SVM** menunjukkan performa terbaik dalam meminimalkan False Positive (hanya 148), namun memiliki False Negative tertinggi (782), yang berarti model ini terlalu konservatif dan melewatkan banyak peluang bisnis. Sebaliknya, **KNN** memiliki False Positive tertinggi (251) dan False Negative yang cukup besar (723), menunjukkan performa yang kurang konsisten dalam kedua kelas. Secara keseluruhan, GBC dan RF merupakan pilihan optimal karena memberikan trade-off terbaik antara menangkap nasabah potensial (True Positive) dan menghindari prediksi yang salah (False Positive), yang sangat penting untuk efisiensi kampanye telemarketing bank.


### 5.5 Feature Importance 

![Feature Importance](https://github.com/adstika20/uts_datascience/blob/main/Image/Feature%20importance.png)

Fitur **duration** (durasi panggilan) merupakan fitur paling dominan dengan importance score mendekati 1.0 pada semua model (SVC, KNN, Random Forest, dan Gradient Boosting), mengindikasikan bahwa durasi interaksi telepon adalah prediktor terkuat untuk keberhasilan kampanye, meskipun fitur ini tidak dapat digunakan untuk prediksi sebelum panggilan dilakukan. Fitur **pdays** (hari sejak kontak terakhir) juga menunjukkan importance tinggi khususnya pada model SVC (~1.0), KNN (~0.67), dan poutcome (~0.88 pada SVC), menandakan bahwa riwayat interaksi sebelumnya sangat berpengaruh terhadap keputusan nasabah. Fitur-fitur **poutcome** (hasil kampanye sebelumnya), **month** (bulan kontak), dan **age** menunjukkan importance moderat (~0.3) yang konsisten across models, mengindikasikan bahwa faktor temporal dan demografis memiliki pengaruh signifikan namun tidak dominan. Secara keseluruhan, untuk keperluan prediksi praktis (tanpa menggunakan duration), kombinasi fitur **poutcome**, **month**, **pdays**, **age**, dan **balance** menjadi prediktor kunci yang harus dioptimalkan dalam strategi telemarketing bank.

---

## 6. Deployment 
Model machine learning yang telah dibangun di-deploy menggunakan **Hugging Face Spaces** dengan framework **Gradio** untuk memudahkan akses dan penggunaan oleh end-user, khususnya manajer kampanye telemarketing bank.

#### 6.1 Persiapan Model dan Artifacts
Sebelum deployment, beberapa file penting dipersiapkan dan disimpan dalam format pickle (.pkl):
- **Model Files**: `gbc_model.pkl`, `knn_model.pkl`, `svm_model.pkl`, `rf_model.pkl` - Model yang telah dilatih
- **Preprocessing Objects**: `scaler.pkl` - StandardScaler untuk normalisasi data input
- **Feature Configuration**: `feature_columns.pkl` - Daftar kolom fitur yang digunakan model
- **Label Encoders**: `label_encoders.pkl` - Encoder untuk fitur kategorikal

#### 6.2 Pengembangan Aplikasi Web (app.py)
Aplikasi Streamlit dikembangkan dengan fitur-fitur utama:
- **Model Selection**: Dropdown untuk memilih model (GBC, KNN, SVM, RF)
- **Input Form**: Form interaktif untuk memasukkan data nasabah (age, job, marital, education, balance, dll.)
- **Prediction Output**: Menampilkan hasil prediksi (Yes/No) dan probability score
- **Visualization**: Menampilkan confidence score dalam bentuk progress bar dan metric cards

#### 6.3 Konfigurasi Dependencies
File `requirements.txt` dibuat dengan library yang diperlukan:
```
streamlit
pandas
numpy
scikit-learn
pickle

```

[Link aplikasi](https://huggingface.co/spaces/atikansh20/bank-term-deposit-prediction)

![](https://github.com/adstika20/uts_datascience/blob/main/Image/Screenshot%20web%20app.png)

---

## Kesimpulan

## 7. Kesimpulan

Penelitian ini berhasil mengembangkan sistem prediksi keberhasilan kampanye telemarketing bank untuk produk deposito jangka panjang menggunakan dataset Bank Marketing. Setelah melalui tahapan preprocessing yang komprehensif dan dilakukan perbandingan empat algoritma machine learning yaitu Random Forest (RF), Support Vector Machine (SVM), Gradient Boosting Classifier (GBC), dan K-Nearest Neighbors (KNN). Hasil evaluasi menunjukkan bahwa **Gradient Boosting Classifier (GBC) merupakan model terbaik** dengan accuracy 91%, precision 0.66, recall 0.43, dan F1-Score 0.52 untuk kelas positif (nasabah berlangganan), mengungguli RF, SVM, dan KNN dalam keseimbangan antara menangkap nasabah potensial dan meminimalkan false positive. Analisis feature importance mengungkapkan bahwa **duration** (durasi panggilan) adalah prediktor terkuat dengan importance score mendekati 1.0 across all models, diikuti oleh **pdays** (hari sejak kontak terakhir), **poutcome** (hasil kampanye sebelumnya), **month**, dan **age** sebagai faktor-faktor kunci yang mempengaruhi keputusan nasabah. Model GBC yang telah dioptimasi berhasil di-deploy melalui platform Hugging Face Spaces menggunakan framework Gradio, menyediakan interface user-friendly yang memungkinkan manajer kampanye untuk melakukan prediksi real-time.

**Namun, penelitian ini memiliki keterbatasan kritis**, terutama pada performa prediksi kelas positif (class 1 - nasabah yang menerima tawaran deposito) yang masih rendah dengan **recall hanya 43%** pada model terbaik GBC. Hal ini mengindikasikan bahwa **lebih dari separuh nasabah potensial (57%) masih terlewatkan** oleh sistem prediksi. Untuk penelitian mendatang, diperlukan penanganan class imbalance yang lebih agresif, eksplorasi advanced algorithms dan historical interaction patterns untuk meningkatkan recall kelas positif minimal ke level 70-80% agar model dapat diimplementasikan secara efektif.

---

## Daftar Pustaka
[1] Feng, L., Zhao, Y., & Zhao, L. (2022). How to improve the success of bank telemarketing? Prediction and interpretability analysis based on machine learning. Computers & Industrial Engineering, 175, 108874. https://doi.org/10.1016/j.cie.2022.108874

[2] Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22-31. https://doi.org/10.1016/j.dss.2014.03.001

[3] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32. DOI: 10.1023/A:1010933404324

[4] Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. Machine Learning, 20(3), 273-297. DOI: 10.1007/BF00994018

[5] Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. The Annals of Statistics, 29(5), 1189-1232. DOI: 10.1214/aos/1013203451

[6] Cover, T., & Hart, P. (1967). Nearest Neighbor Pattern Classification. IEEE Transactions on Information Theory, 13(1), 21-27. DOI: 10.1109/TIT.1967.1053964




---

