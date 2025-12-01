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
1. Mengembangkan model prediksi diabetes dengan menerapkan pipeline preprocessing yang mencegah data leakage pada dataset PIMA
2. Membangun model prediktif yang akurat untuk memprediksi keberhasilan telemarketing
3. Mengidentifikasi faktor-faktor kunci yang mempengaruhi keputusan nasabah
4. Membandingkan performa berbagai algoritma machine learning (Random Forest, Support Vector Machine, K-Nearest Neighbors, Gradient Boosting)
5. Mengembangkan web application di Platform Hugging Face untuk deployment model

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

| Logistic Regression | Neural Network (MLP) |
|---------------------|-----------------------|
| ![LR](https://raw.githubusercontent.com/adstika20/datascience_proyek/main/image/LR.png) | ![MLP](https://raw.githubusercontent.com/adstika20/datascience_proyek/main/image/NN%20MLP.png) |

| Random Forest | SVM |
|---------------|-----|
| ![RF](https://raw.githubusercontent.com/adstika20/datascience_proyek/main/image/RF.png) | ![SVM](https://raw.githubusercontent.com/adstika20/datascience_proyek/main/image/SVM.png) |

Secara keseluruhan, keempat confusion matrix memperlihatkan pola yang konsisten bahwa seluruh model memiliki kapasitas klasifikasi yang jauh lebih kuat terhadap kelas 0 (negatif diabetes) dibandingkan kelas 1 (positif diabetes). Logistic Regression dan Random Forest menunjukkan kinerja yang relatif stabil dengan kemampuan identifikasi kasus positif yang moderat namun masih dibayangi tingkat false negative yang substansial. SVM menampilkan performa paling seimbang di antara model konvensional, ditandai oleh false positive yang rendah dan kemampuan deteksi kelas positif yang sedikit lebih baik, meskipun kecenderungan bias menuju kelas mayoritas tetap dominan. Sebaliknya, MLP memperlihatkan kegagalan total dalam mengenali kelas positif karena seluruh instans kelas 1 diklasifikasikan sebagai 0, menandakan ketidakmampuan model dalam mempelajari representasi kelas minoritas.

### 5.5 Feature Importance dan Permutation 

#### Perbandingan Feature Importance Antar Model

| Model                | Glucose | BMI      | Age      | Pregnancies |
|----------------------|---------|----------|----------|-------------|
| **Logistic Regression** | **1.1060** | 0.5350   | 0.0801   | 0.5133      |
| **SVM (RBF)**           | **0.9221** | 0.3303   | 0.0212   | 0.4064      |
| **Random Forest**       | **0.3850** | 0.2589   | 0.2247   | 0.1314      |
| **Neural Network (MLP)** (Permutation) | **0.0969** | -0.0242  | 0.0008   | 0.0016      |

**Glucose** konsisten menjadi fitur dengan pengaruh terbesar pada tiga model utama(LR, SVM, dan RF), menegaskan bahwa fitur ini merupakan sinyal paling kuat dalam memprediksi diabetes. **BMI** dan **Pregnancies** muncul dengan kontribusi menengah, sementara **Age** hanya menunjukkan pengaruh moderat pada **RF** dan hampir tidak signifikan pada model lainnya. **MLP** tidak memiliki feature importance internal sehingga dihitung menggunakan Permutation Importance. Hasilnya menunjukkan nilai importance yang sangat kecil dan tidak stabil, menandakan bahwa mengacak fitur-fitur tersebut tidak menurunkan performa model, sehingga MLP memang gagal mempelajari hubungan prediktif di data.

---

## 6. Deployment 
Proyek ini dideploy menggunakan Hugging Face Spaces dengan framework Gradio untuk menyediakan antarmuka prediksi diabetes berbasis model machine learning yang telah dilatih.
Tahapan Deployment
- Menyiapkan model dan scaler '.pkl' (LR, SVM, RF, NN) dan scaler.pkl diunggah ke Hugging Face.
- Membuat aplikasi Gradio (app.py)
- Menambahkan requirements.txt
- Push semua file ke Hugging Face
Build dilakukan otomatis, lalu Space langsung aktif dengan URL publik.

[Link aplikasi](https://huggingface.co/spaces/atikansh20/pima-diabetes-classifier)

![](https://github.com/adstika20/datascience_proyek/blob/main/image/Hasil%20prediksi.png)

---

## Kesimpulan

## Kesimpulan

Diabetes melitus dengan angka kematian global 1,6 juta jiwa per tahun menuntut sistem deteksi dini yang lebih efektif, terutama melalui pendekatan berbasis data menggunakan variabel klinis sederhana. Penelitian ini mengembangkan model prediksi diabetes pada dataset PIMA dengan mengatasi kelemahan metodologis studi sebelumnya melalui pipeline preprocessing yang mencegah data leakage dan evaluasi metrik klinis komprehensif. Setelah feature selection menghasilkan 4 fitur optimal (Glucose, BMI, Age, Pregnancies) dan perbandingan empat algoritma via 10-fold cross-validation, **Logistic Regression terbukti paling optimal** dengan F1-score 0,5938, accuracy 78,45%, dan interpretability superior, dimana **Glucose konsisten menjadi prediktor terkuat**. Namun, **sensitivity 51,79% menunjukkan kelemahan kritis** karena hampir setengah pasien diabetes tidak terdeteksi, sehingga model belum layak untuk implementasi klinis tanpa optimasi lanjutan seperti threshold tuning atau penanganan class imbalance untuk meningkatkan sensitivity.

---

## Kesimpulan

Diabetes melitus dengan angka kematian global 1,6 juta jiwa per tahun menuntut sistem deteksi dini yang lebih efektif, terutama melalui pendekatan berbasis data menggunakan variabel klinis sederhana. Penelitian ini mengembangkan model prediksi diabetes pada dataset PIMA dengan mengatasi kelemahan metodologis studi sebelumnya melalui pipeline preprocessing yang mencegah data leakage dan evaluasi metrik klinis komprehensif. Setelah feature selection menghasilkan 4 fitur optimal (Glucose, BMI, Age, Pregnancies) dan perbandingan empat algoritma via 10-fold cross-validation, **Logistic Regression terbukti paling optimal** dengan F1-score 0,5938, accuracy 78,45%, dan interpretability superior, dimana **Glucose konsisten menjadi prediktor terkuat**. Namun, **sensitivity 51,79% menunjukkan kelemahan kritis** karena hampir setengah pasien diabetes tidak terdeteksi, sehingga model belum layak untuk implementasi klinis tanpa optimasi lanjutan seperti threshold tuning atau penanganan class imbalance untuk meningkatkan sensitivity.

---

## Daftar Pustaka
[1] Feng, L., Zhao, Y., & Zhao, L. (2022). How to improve the success of bank telemarketing? Prediction and interpretability analysis based on machine learning. Computers & Industrial Engineering, 175, 108874. https://doi.org/10.1016/j.cie.2022.108874

[2] Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22-31. https://doi.org/10.1016/j.dss.2014.03.001

[3] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32. DOI: 10.1023/A:1010933404324

[4] Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. Machine Learning, 20(3), 273-297. DOI: 10.1007/BF00994018

[5] Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. The Annals of Statistics, 29(5), 1189-1232. DOI: 10.1214/aos/1013203451

[6] Cover, T., & Hart, P. (1967). Nearest Neighbor Pattern Classification. IEEE Transactions on Information Theory, 13(1), 21-27. DOI: 10.1109/TIT.1967.1053964




---

