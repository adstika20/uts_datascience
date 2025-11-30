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

Korelasi menunjukkan bahwa `Glucose` memiliki hubungan paling kuat dengan `Outcome` sehingga menjadi indikator utama risiko diabetes. Variabel seperti `BMI`, `Age`, dan `Insulin` memiliki korelasi sedang dengan `Outcome`. Sementara itu, fitur lain memiliki korelasi rendah satu sama lain sehingga tidak terdapat multikolinearitas berarti. 

## 3. Data Preparation  

### 3.1 Feature Selection Based on Correlation
**Metode:** Pearson correlation coefficient dengan threshold 0.2

- Mengurangi dimensionalitas dengan mempertahankan hanya fitur yang memiliki korelasi signifikan dengan target (Outcome)
- Threshold 0.2 dipilih sebagai cut-off konservatif untuk memastikan fitur yang dipertahankan memiliki hubungan minimal dengan diabetes
- Menghindari multicollinearity dan overfitting pada model sederhana

Berdasarkan analisis korelasi, 4 fitur terpilih dengan correlation > 0.2:

| Feature | Correlation with Outcome | Interpretasi Klinis |
|---------|--------------------------|---------------------|
| Glucose | 0.48 | Kadar glukosa darah (prediktor terkuat) |
| BMI | 0.32 | Indeks massa tubuh (faktor risiko obesitas) |
| Age | 0.24 | Usia (risiko meningkat seiring usia) |
| Pregnancies | 0.22 | Riwayat kehamilan (gestational diabetes risk) |

**Fitur yang Di-drop:**
- **Insulin** (corr: 0.15) - Korelasi lemah + 48% missing values
- **SkinThickness** (corr: 0.07) - Korelasi sangat lemah + 30% missing values
- **BloodPressure** (corr: 0.18) - Di bawah threshold
- **DiabetesPedigreeFunction** (corr: 0.17) - Di bawah threshold

**Hasil:**
- **Original:** 8 features → **Selected:** 4 features
- **Dataset shape:** 636 samples × 4 features + 1 target
- **Dimensionality reduction:** 50%

### 3.2 Data Standardization

**Metode:** StandardScaler (Z-score normalization)

**Formula:**
```
z = (x - μ) / σ
```
Dimana:
- μ = mean
- σ = standard deviation

**Implementasi:**
```python
Pipeline Structure:
1. StandardScaler  → Fit pada training data, transform pada train dan test
2. Model           → Training dengan data yang sudah di-scale
```

**⚠️ CRITICAL: Pencegahan Data Leakage**
- StandardScaler **hanya di-fit pada training data**
- Test data **hanya di-transform** menggunakan parameter (mean, std) dari training data
- Implementasi menggunakan **sklearn Pipeline** untuk memastikan proper scaling workflow
- Scaling dilakukan **SETELAH** train-test split, **BUKAN SEBELUM**

**Alasan Tidak Pakai Normalization (Min-Max Scaling):**
- StandardScaler lebih robust terhadap outliers
- Tidak mengasumsikan distribusi bounded (0-1)
- Lebih cocok untuk algoritma linear yang mengasumsikan distribusi normal
---

## 4. Modeling  

#### **4.1 Logistic Regression(LR)**

LR diterapkan sebagai model baseline karena sifatnya yang linear, interpretatif, dan sesuai untuk hubungan biner antara faktor klinis dan status diabetes [[5]](https://doi.org/10.30871/jaic.v9i5.9815). Model ini menggunakan solver `lbfgs` untuk optimasi pada dataset berdimensi rendah seperti PIMA, dengan batas iterasi diperpanjang hingga 2000 untuk memastikan konvergensi penuh. Regularisasi tetap aktif melalui konfigurasi default sehingga koefisien lebih stabil terhadap multikolinearitas pada fitur seperti `Glucose`, `BMI`, dan `Age`. Penggunaan `random_state` memastikan reprodusibilitas, sedangkan formulasi logit memberikan probabilitas kejadian diabetes yang secara medis relevan untuk menilai risiko. 

#### **2. Random Forest Classifier (RF)**
RF digunakan sebagai pendekatan ensemble berbasis pohon yang mampu menangani interaksi kompleks antar fitur klinis tanpa asumsi linearitas [[6]](https://jurnal.ipdig.id/index.php/jtid/article/view/41). Model dikonfigurasi dengan 100 pohon untuk mencapai kinerja dan efisiensi komputasi, sementara max_depth=`None` memungkinkan setiap pohon tumbuh secara penuh sehingga pola non-linear dapat dieksplorasi secara maksimal. Nilai min_samples_split=`2` mempertahankan sensitivitas model terhadap variasi kasus minoritas, yang penting karena kelas positif (diabetes) pada dataset PIMA relatif lebih sedikit. Parameter `random_state` digunakan untuk memastikan reprodusibilitas struktur hutan. 

#### **3. Support Vector Machine (SVM)**
SVM dengan kernel RBF digunakan untuk menangkap hubungan non-linear antara variabel klinis[[7]](https://doi.org/10.3390/info15040235). Konfigurasi kernel RBF dengan gamma=`scale` memungkinkan model menyesuaikan tingkat kepekaan terhadap variasi fitur tanpa risiko overfitting yang ekstrem. Parameter probability=`True` diaktifkan agar model menghasilkan probabilitas kelas. Penetapan `random_state` menjaga konsistensi hasil. 

#### **4. Multilayer Perceptron (MLP Neural Network)**
MLP dibangun dengan arsitektur dua lapisan tersembunyi berukuran 26 dan 5 unit, yang disesuaikan dengan jumlah fitur dan kompleksitas pola pada dataset PIMA. Aktivasi ReLU digunakan untuk menjaga stabilitas gradien, sementara solver=`sgd` dengan learning_rate=`adaptive` memungkinkan penyesuaian laju pembelajaran ketika performa validasi stagnan. Parameter seperti learning_rate_init=`0.01`, momentum=`0.9`, dan batch_size=`32` memastikan proses pelatihan berlangsung stabil pada data berskala kecil. Mekanisme early_stopping dengan validation_fraction=`0.1` menghentikan pelatihan ketika model tidak lagi menunjukkan peningkatan berarti selama 50 iterasi, sehingga mengurangi risiko overfitting. Penetapan max_iter=`800` memberi ruang cukup untuk konvergensi. Model ini dipilih karena mampu menangkap hubungan non-linear multivariabel sekaligus memberikan fleksibilitas untuk mendeteksi pola interaksi yang tidak tertangkap oleh model linear[[8]](https://doi.org/10.3390/electronics8020194).

---

## 5. Evaluation  

### 5.1 Metode Evaluasi

Model dievaluasi menggunakan **10-Fold Stratified Cross-Validation**, di mana setiap fold mempertahankan proporsi kelas diabetes dan non-diabetes agar estimasi performa lebih stabil dan mengurangi bias. 
- Accuracy mengukur proporsi prediksi benar secara keseluruhan..
- Precision menunjukkan proporsi prediksi positif (pasien diabetes) yang benar, membantu menghindari false alarm.
- Sensitivity (Recall kelas positif) menilai kemampuan model mendeteksi pasien diabetes, yang sangat penting untuk meminimalkan pasien yang terlewatkan.
- Specificity (Recall kelas negatif) mengukur kemampuan model mengenali pasien non-diabetes, mencegah overdiagnosis dan beban psikologis yang tidak perlu. 
- F1-Score, sebagai harmonic mean dari precision dan recall, memberikan keseimbangan antara deteksi pasien positif dan minimisasi kesalahan prediksi.
Semua metrik ini dihitung rata-rata dari 10 fold, sehingga hasil evaluasi lebih robust dan mewakili performa model secara keseluruhan.

### 5.3 Hasil Perbandingan Model (10-Fold CV)

| Model                | Accuracy | Precision | Sensitivity | Specificity | F1-Score |
|------------------------|-------------|--------------|----------------|----------------|-------------|
| **Logistic Regression** | ⭐ **0.7845** | 0.7228       | 0.5179         | 0.9043         | ⭐ **0.5938** |
| **SVM (RBF)**           | 0.7830      | ⭐ **0.7267** | 0.5079         | ⭐ **0.9067**   | 0.5869       |
| **Random Forest**       | 0.7610      | 0.6590       | ⭐ **0.5284**   | 0.8657         | 0.5766       |
| **Neural Network (MLP)**| 0.7751      | 0.6996       | 0.5016         | 0.8975         | 0.5757       |

Keempat model menunjukkan pola yang konsisten: **specificity tinggi (86–91%)** namun **sensitivity rendah (50–53%)**. Artinya, semua model cukup baik mengenali pasien sehat (kelas 0) tetapi lemah mendeteksi pasien diabetes (kelas 1), sehingga hampir setengah pasien diabetes tidak teridentifikasi oleh model terbaik sekalipun. **LR** memberikan keseimbangan terbaik antara precision dan recall, dengan akurasi 78,45% dan F1-score 0,5938. Model ini cukup baik mendeteksi pasien positif tanpa mengorbankan terlalu banyak pasien sehat. Namun, sensitivity 51,79% menunjukkan masih ada hampir setengah pasien diabetes yang tidak terdeteksi, sehingga tetap memerlukan verifikasi klinis.

Sementara itu, **SVM** menunjukkan performa berbeda. Precision tertinggi (72,67%) dan specificity 90,67% menandakan prediksi positif sangat dapat diandalkan, namun hal ini dicapai dengan mengorbankan sensitivity (50,79%). Artinya, SVM terlalu konservatif dalam menandai pasien diabetes. Dibanding LR, SVM lebih aman untuk pasien sehat tetapi kurang optimal untuk skrining awal yang membutuhkan deteksi sebanyak mungkin kasus positif.

**RF** menunjukkan **sensitivity** tertinggi (52,84%), model ini mampu menangkap lebih banyak kasus positif dibanding model lain, meskipun meningkatkan jumlah false positive 95 pasien sehat salah diklasifikasikan dibanding 70 pada Logistic Regression. Terakhir, **Neural Network (MLP)** memiliki performa rata-rata pada semua metrik dan sensitivity rendah (50,16%) menunjukkan kompleksitas MLP tidak memberi manfaat pada dataset kecil ini (768 sampel, 8 fitur). 

Secara keseluruhan, **LR** merupakan model paling optimal dengan F1-score tertinggi (0,5938) dan accuracy terbaik (0,7845), menunjukkan keseimbangan terbaik antara precision dan recall dibanding model lainnya. Kelebihan tambahan LR adalah interpretability yang memudahkan penjelasan ke klinisi serta kesederhanaan model yang menghindari risiko overfitting pada dataset berukuran kecil (768 sampel). 

Hasil ini juga menunjukkan implikasi penting terkait penelitian Khanam & Foo (2021), yang melaporkan Neural Network dengan akurasi 88,6% namun tanpa transparansi sensitivity/specificity.Penelitian ini hanya (78.45%) lebih dapat dipercaya karena: (1) preprocessing dilakukan setelah data splitting untuk mencegah leakage, (2) evaluasi menggunakan 5 metrik yang komprehensif, dan (3) mengungkap kelemahan kritis yang tidak terlihat jika hanya fokus pada accuracy. Accuracy tinggi tanpa sensitivity memadai menunjukkan model hanya pintar menebak kelas mayoritas—fenomena yang tersembunyi dalam evaluasi penelitian sebelumnya.


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

## Daftar Pustaka
[1] World Health Organization, “Diabetes,” 2024. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/diabetes

[2] X. Lin, Y. Xu, X. Pan, J. Xu, Y. Ding, X. Sun, X. Song, Y. Ren, and P.-F. Shan, "Global, regional, and national burden and trend of diabetes in 195 countries and territories: an analysis from 1990 to 2025," *Scientific Reports*, vol. 10, p. 14790, Sep. 2020, doi: 10.1038/s41598-020-71908-9.

[3] J. J. Khanam and S. Y. Foo, "A comparison of machine learning algorithms for diabetes prediction," *ICT Express*, vol. 7, no. 4, pp. 432–439, 2021, doi: 10.1016/j.icte.2021.02.004.

[4] J. Shreffler and M. R. Huecker, "Diagnostic Testing Accuracy: Sensitivity, Specificity, Predictive Values and Likelihood Ratios," in *StatPearls* [Internet]. Treasure Island, FL, USA: StatPearls Publishing, 2025 Jan–. Updated Mar. 6, 2023. Available: https://www.ncbi.nlm.nih.gov/books/NBK557491/

[5] M. F. Kurniawan and D. A. Megawaty, "Comparison of Logistic Regression, Random Forest, Support Vector Machine (SVM) and K-Nearest Neighbor (KNN) Algorithms in Diabetes Prediction," *Journal of Applied Informatics and Computing*, vol. 9, no. 5, Oct. 2025, doi: 10.30871/jaic.v9i5.9815.

[6] B. Siswoyo and M. I. Nurhafidz, "Penerapan Algoritma Random Forest Untuk Prediksi Risiko Diabetes Berdasarkan Data Kesehatan Pasien," *Jurnal Teknologi Informasi Digital*, vol. 1, no. 1, 2025.

[7] R. Guido, S. Ferrisi, D. Lofaro, and D. Conforti, "An overview on the advancements of support vector machine models in healthcare applications: a review," Information, vol. 15, no. 4, p. 235, Apr. 2024, doi: 10.3390/info15040235.

[8] I. U. Rehman, M. M. Nasralla, and N. Y. Philip, "Multilayer perceptron neural network-based QoS-aware, content-aware and device-aware QoE prediction model: A proposed prediction model for medical ultrasound streaming over small cell networks," Electronics, vol. 8, no. 2, p. 194, Feb. 2019, doi: 10.3390/electronics8020194.


---



Selain itu, evaluasi performa model terlalu berfokus pada akurasi tanpa mempertimbangkan sensitivitas, spesifisitas, dan confusion matrix yang sangat penting dalam konteks data medis. Dalam aplikasi klinis, sensitivitas tinggi diperlukan untuk meminimalkan false negative yang dapat menyebabkan pasien diabetes tidak terdeteksi, sementara spesifisitas penting untuk menghindari false positive yang mengakibatkan overdiagnosis dan beban psikologis pasien[[4]](https://www.ncbi.nlm.nih.gov/books/NBK557491/). Oleh karena itu, penelitian ini bertujuan mengatasi keterbatasan metodologis tersebut dengan menerapkan pipeline preprocessing yang mencegah data leakage, evaluasi metrik yang komprehensif sesuai standar klinis, serta eksplorasi teknik feature engineering untuk mengoptimalkan performa prediksi diabetes.

### 1.2 Rumusan Masalah
Bagaimana membangun model machine learning yang valid, terukur, dan dapat direplikasi untuk memprediksi risiko diabetes berdasarkan dataset PIMA, dengan evaluasi yang mencakup aspek medis (sensitivity dan specificity), serta menerapkan proses preprocessing dan cross-validation yang benar sehingga menghasilkan model yang komprehensif untuk mendukung proses deteksi dini di sektor kesehatan?

### 1.3 Tujuan 
- Mengembangkan model prediksi diabetes dengan menerapkan pipeline preprocessing yang mencegah data leakage pada dataset PIMA
- Membandingkan performa berbagai algoritma machine learning menggunakan metrik evaluasi klinis yang komprehensif
- Mengidentifikasi fitur klinis yang paling berkontribusi terhadap prediksi diabetes melalui analisis feature importance.
---

## 2. Data Understanding  

### 2.1 Deskripsi dataset

**Nama:** [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
**Sumber:** National Institute of Diabetes and Digestive and Kidney Diseases  
**Jumlah sampel:** 768 

#### Variabel Fitur

| Fitur         | Deskripsi                                                                            | Tipe Data     | Rata-rata(Mean) |
|-----------------|---------------------------------------------------------------------------------------------------------|----------|-----------|
| Pregnancies            | Jumlah total kehamilan yang pernah dialami oleh pasien.                                                | Numerik  | 3.85      |
| Glucose         | Konsentrasi glukosa plasma 2 jam setelah tes toleransi glukosa oral.                                   | Numerik  | 120.89    |
| BloodPressure              | Tekanan darah diastolik (mm Hg) saat pemeriksaan.                                                      | Numerik  | 69.11     |
| SkinThickness   | Tebal lipatan kulit trisep (mm) sebagai indikator lemak subkutan.                                      | Numerik  | 20.54     |
| Insulin         | Kadar insulin serum 2 jam setelah konsumsi glukosa (µIU/mL).                                           | Numerik  | 79.80     |
| BMI             | Indeks massa tubuh (kg/m²), indikator berat badan relatif terhadap tinggi badan.                       | Numerik  | 32        |
| DiabetesPedigreeFunction	             | Fungsi silsilah diabetes; mengukur predisposisi genetik terhadap diabetes berdasarkan riwayat keluarga. | Numerik  | 0.47      |
| Age             | Usia pasien dalam tahun.                                                                               | Numerik  | 33        |
| Outcome         | Hasil diagnosis diabetes (1 = positif diabetes, 0 = negatif diabetes).                                 | Nominal  | –         |

#### Variabel Target

Dataset ini bertujuan memprediksi status diabetes pasien (`Outcome`) berdasarkan 8 variabel klinis. Berikut distribusi untuk kelas `Outcome`

| Kelas | Jumlah | Persentase | Visualisasi |
|-------|--------|------------|-------------|
| 0 (Negatif) | 500 | 65.1% | ████████████████ |
| 1 (Positif) | 268 | 34.9% | ████████ |
| **Total** | **768** | **100%** | |

### 2.1 Eksplorasi Data

#### Univariate Analysis

| Pregnancies | Glucose | Blood Pressure | Skin Thickness |
|------------|---------|----------------|----------------|
| ![](https://raw.githubusercontent.com/adstika20/datascience_proyek/main/image/pregnancies.png) | ![](https://raw.githubusercontent.com/adstika20/datascience_proyek/main/image/glucose.png) | ![](https://raw.githubusercontent.com/adstika20/datascience_proyek/main/image/blood.png) | ![](https://raw.githubusercontent.com/adstika20/datascience_proyek/main/image/skin.png) |

| Insulin | BMI | Diabetes Pedigree | Age |
|---------|-----|-------------------|-----|
| ![](https://raw.githubusercontent.com/adstika20/datascience_proyek/main/image/insulin.png) | ![](https://raw.githubusercontent.com/adstika20/datascience_proyek/main/image/bmi.png) | ![](https://raw.githubusercontent.com/adstika20/datascience_proyek/main/image/pdf.png) | ![](https://raw.githubusercontent.com/adstika20/datascience_proyek/main/image/age.png) |

Sebagian fitur, seperti `Glucose`, `BMI`, dan `BloodPressure`, memiliki distribusi relatif normal sehingga dapat diandalkan dalam modeling, meskipun nilai 0 pada `Glucose` dan `BloodPressure` tetap perlu diperlakukan sebagai missing. `Pregnancies`, `DiabetesPedigreeFunction`, dan `Age` cenderung right-skewed sehingga memerlukan transformasi atau strategi pemodelan khusus, sementara `SkinThickness` dan `Insulin` menghadapi isu serius karena tingginya proporsi nilai 0 yang jelas mencerminkan missing values, masing-masing sekitar 30 persen dan 48 persen. Kondisi ini menyebabkan kedua fitur tersebut berpotensi menurunkan performa model jika tidak ditangani melalui imputasi lanjutan atau bahkan dikeluarkan dari analisis.
  
#### Corellation Analysis
![Korelasi Antar Variabel](https://raw.githubusercontent.com/adstika20/datascience_proyek/main/image/korelasi%20antar%20variabel.png)

Korelasi menunjukkan bahwa `Glucose` memiliki hubungan paling kuat dengan `Outcome` sehingga menjadi indikator utama risiko diabetes. Variabel seperti `BMI`, `Age`, dan `Insulin` memiliki korelasi sedang dengan `Outcome`. Sementara itu, fitur lain memiliki korelasi rendah satu sama lain sehingga tidak terdapat multikolinearitas berarti. 

## 3. Data Preparation  

### 3.1 Feature Selection Based on Correlation
**Metode:** Pearson correlation coefficient dengan threshold 0.2

- Mengurangi dimensionalitas dengan mempertahankan hanya fitur yang memiliki korelasi signifikan dengan target (Outcome)
- Threshold 0.2 dipilih sebagai cut-off konservatif untuk memastikan fitur yang dipertahankan memiliki hubungan minimal dengan diabetes
- Menghindari multicollinearity dan overfitting pada model sederhana

Berdasarkan analisis korelasi, 4 fitur terpilih dengan correlation > 0.2:

| Feature | Correlation with Outcome | Interpretasi Klinis |
|---------|--------------------------|---------------------|
| Glucose | 0.48 | Kadar glukosa darah (prediktor terkuat) |
| BMI | 0.32 | Indeks massa tubuh (faktor risiko obesitas) |
| Age | 0.24 | Usia (risiko meningkat seiring usia) |
| Pregnancies | 0.22 | Riwayat kehamilan (gestational diabetes risk) |

**Fitur yang Di-drop:**
- **Insulin** (corr: 0.15) - Korelasi lemah + 48% missing values
- **SkinThickness** (corr: 0.07) - Korelasi sangat lemah + 30% missing values
- **BloodPressure** (corr: 0.18) - Di bawah threshold
- **DiabetesPedigreeFunction** (corr: 0.17) - Di bawah threshold

**Hasil:**
- **Original:** 8 features → **Selected:** 4 features
- **Dataset shape:** 636 samples × 4 features + 1 target
- **Dimensionality reduction:** 50%

### 3.2 Data Standardization

**Metode:** StandardScaler (Z-score normalization)

**Formula:**
```
z = (x - μ) / σ
```
Dimana:
- μ = mean
- σ = standard deviation

**Implementasi:**
```python
Pipeline Structure:
1. StandardScaler  → Fit pada training data, transform pada train dan test
2. Model           → Training dengan data yang sudah di-scale
```

**⚠️ CRITICAL: Pencegahan Data Leakage**
- StandardScaler **hanya di-fit pada training data**
- Test data **hanya di-transform** menggunakan parameter (mean, std) dari training data
- Implementasi menggunakan **sklearn Pipeline** untuk memastikan proper scaling workflow
- Scaling dilakukan **SETELAH** train-test split, **BUKAN SEBELUM**

**Alasan Tidak Pakai Normalization (Min-Max Scaling):**
- StandardScaler lebih robust terhadap outliers
- Tidak mengasumsikan distribusi bounded (0-1)
- Lebih cocok untuk algoritma linear yang mengasumsikan distribusi normal
---

## 4. Modeling  

#### **4.1 Logistic Regression(LR)**

LR diterapkan sebagai model baseline karena sifatnya yang linear, interpretatif, dan sesuai untuk hubungan biner antara faktor klinis dan status diabetes [[5]](https://doi.org/10.30871/jaic.v9i5.9815). Model ini menggunakan solver `lbfgs` untuk optimasi pada dataset berdimensi rendah seperti PIMA, dengan batas iterasi diperpanjang hingga 2000 untuk memastikan konvergensi penuh. Regularisasi tetap aktif melalui konfigurasi default sehingga koefisien lebih stabil terhadap multikolinearitas pada fitur seperti `Glucose`, `BMI`, dan `Age`. Penggunaan `random_state` memastikan reprodusibilitas, sedangkan formulasi logit memberikan probabilitas kejadian diabetes yang secara medis relevan untuk menilai risiko. 

#### **2. Random Forest Classifier (RF)**
RF digunakan sebagai pendekatan ensemble berbasis pohon yang mampu menangani interaksi kompleks antar fitur klinis tanpa asumsi linearitas [[6]](https://jurnal.ipdig.id/index.php/jtid/article/view/41). Model dikonfigurasi dengan 100 pohon untuk mencapai kinerja dan efisiensi komputasi, sementara max_depth=`None` memungkinkan setiap pohon tumbuh secara penuh sehingga pola non-linear dapat dieksplorasi secara maksimal. Nilai min_samples_split=`2` mempertahankan sensitivitas model terhadap variasi kasus minoritas, yang penting karena kelas positif (diabetes) pada dataset PIMA relatif lebih sedikit. Parameter `random_state` digunakan untuk memastikan reprodusibilitas struktur hutan. 

#### **3. Support Vector Machine (SVM)**
SVM dengan kernel RBF digunakan untuk menangkap hubungan non-linear antara variabel klinis[[7]](https://doi.org/10.3390/info15040235). Konfigurasi kernel RBF dengan gamma=`scale` memungkinkan model menyesuaikan tingkat kepekaan terhadap variasi fitur tanpa risiko overfitting yang ekstrem. Parameter probability=`True` diaktifkan agar model menghasilkan probabilitas kelas. Penetapan `random_state` menjaga konsistensi hasil. 

#### **4. Multilayer Perceptron (MLP Neural Network)**
MLP dibangun dengan arsitektur dua lapisan tersembunyi berukuran 26 dan 5 unit, yang disesuaikan dengan jumlah fitur dan kompleksitas pola pada dataset PIMA. Aktivasi ReLU digunakan untuk menjaga stabilitas gradien, sementara solver=`sgd` dengan learning_rate=`adaptive` memungkinkan penyesuaian laju pembelajaran ketika performa validasi stagnan. Parameter seperti learning_rate_init=`0.01`, momentum=`0.9`, dan batch_size=`32` memastikan proses pelatihan berlangsung stabil pada data berskala kecil. Mekanisme early_stopping dengan validation_fraction=`0.1` menghentikan pelatihan ketika model tidak lagi menunjukkan peningkatan berarti selama 50 iterasi, sehingga mengurangi risiko overfitting. Penetapan max_iter=`800` memberi ruang cukup untuk konvergensi. Model ini dipilih karena mampu menangkap hubungan non-linear multivariabel sekaligus memberikan fleksibilitas untuk mendeteksi pola interaksi yang tidak tertangkap oleh model linear[[8]](https://doi.org/10.3390/electronics8020194).

---

## 5. Evaluation  

### 5.1 Metode Evaluasi

Model dievaluasi menggunakan **10-Fold Stratified Cross-Validation**, di mana setiap fold mempertahankan proporsi kelas diabetes dan non-diabetes agar estimasi performa lebih stabil dan mengurangi bias. 
- Accuracy mengukur proporsi prediksi benar secara keseluruhan..
- Precision menunjukkan proporsi prediksi positif (pasien diabetes) yang benar, membantu menghindari false alarm.
- Sensitivity (Recall kelas positif) menilai kemampuan model mendeteksi pasien diabetes, yang sangat penting untuk meminimalkan pasien yang terlewatkan.
- Specificity (Recall kelas negatif) mengukur kemampuan model mengenali pasien non-diabetes, mencegah overdiagnosis dan beban psikologis yang tidak perlu. 
- F1-Score, sebagai harmonic mean dari precision dan recall, memberikan keseimbangan antara deteksi pasien positif dan minimisasi kesalahan prediksi.
Semua metrik ini dihitung rata-rata dari 10 fold, sehingga hasil evaluasi lebih robust dan mewakili performa model secara keseluruhan.

### 5.3 Hasil Perbandingan Model (10-Fold CV)

| Model                | Accuracy | Precision | Sensitivity | Specificity | F1-Score |
|------------------------|-------------|--------------|----------------|----------------|-------------|
| **Logistic Regression** | ⭐ **0.7845** | 0.7228       | 0.5179         | 0.9043         | ⭐ **0.5938** |
| **SVM (RBF)**           | 0.7830      | ⭐ **0.7267** | 0.5079         | ⭐ **0.9067**   | 0.5869       |
| **Random Forest**       | 0.7610      | 0.6590       | ⭐ **0.5284**   | 0.8657         | 0.5766       |
| **Neural Network (MLP)**| 0.7751      | 0.6996       | 0.5016         | 0.8975         | 0.5757       |

Keempat model menunjukkan pola yang konsisten: **specificity tinggi (86–91%)** namun **sensitivity rendah (50–53%)**. Artinya, semua model cukup baik mengenali pasien sehat (kelas 0) tetapi lemah mendeteksi pasien diabetes (kelas 1), sehingga hampir setengah pasien diabetes tidak teridentifikasi oleh model terbaik sekalipun. **LR** memberikan keseimbangan terbaik antara precision dan recall, dengan akurasi 78,45% dan F1-score 0,5938. Model ini cukup baik mendeteksi pasien positif tanpa mengorbankan terlalu banyak pasien sehat. Namun, sensitivity 51,79% menunjukkan masih ada hampir setengah pasien diabetes yang tidak terdeteksi, sehingga tetap memerlukan verifikasi klinis.

Sementara itu, **SVM** menunjukkan performa berbeda. Precision tertinggi (72,67%) dan specificity 90,67% menandakan prediksi positif sangat dapat diandalkan, namun hal ini dicapai dengan mengorbankan sensitivity (50,79%). Artinya, SVM terlalu konservatif dalam menandai pasien diabetes. Dibanding LR, SVM lebih aman untuk pasien sehat tetapi kurang optimal untuk skrining awal yang membutuhkan deteksi sebanyak mungkin kasus positif.

**RF** menunjukkan **sensitivity** tertinggi (52,84%), model ini mampu menangkap lebih banyak kasus positif dibanding model lain, meskipun meningkatkan jumlah false positive 95 pasien sehat salah diklasifikasikan dibanding 70 pada Logistic Regression. Terakhir, **Neural Network (MLP)** memiliki performa rata-rata pada semua metrik dan sensitivity rendah (50,16%) menunjukkan kompleksitas MLP tidak memberi manfaat pada dataset kecil ini (768 sampel, 8 fitur). 

Secara keseluruhan, **LR** merupakan model paling optimal dengan F1-score tertinggi (0,5938) dan accuracy terbaik (0,7845), menunjukkan keseimbangan terbaik antara precision dan recall dibanding model lainnya. Kelebihan tambahan LR adalah interpretability yang memudahkan penjelasan ke klinisi serta kesederhanaan model yang menghindari risiko overfitting pada dataset berukuran kecil (768 sampel). 

Hasil ini juga menunjukkan implikasi penting terkait penelitian Khanam & Foo (2021), yang melaporkan Neural Network dengan akurasi 88,6% namun tanpa transparansi sensitivity/specificity.Penelitian ini hanya (78.45%) lebih dapat dipercaya karena: (1) preprocessing dilakukan setelah data splitting untuk mencegah leakage, (2) evaluasi menggunakan 5 metrik yang komprehensif, dan (3) mengungkap kelemahan kritis yang tidak terlihat jika hanya fokus pada accuracy. Accuracy tinggi tanpa sensitivity memadai menunjukkan model hanya pintar menebak kelas mayoritas—fenomena yang tersembunyi dalam evaluasi penelitian sebelumnya.


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

## Daftar Pustaka
[1] Feng, L., Zhao, Y., & Zhao, L. (2022). How to improve the success of bank telemarketing? Prediction and interpretability analysis based on machine learning. Computers & Industrial Engineering, 175, 108874. https://doi.org/10.1016/j.cie.2022.108874

[2] Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22-31. https://doi.org/10.1016/j.dss.2014.03.001

[3] J. J. Khanam and S. Y. Foo, "A comparison of machine learning algorithms for diabetes prediction," *ICT Express*, vol. 7, no. 4, pp. 432–439, 2021, doi: 10.1016/j.icte.2021.02.004.

[4] J. Shreffler and M. R. Huecker, "Diagnostic Testing Accuracy: Sensitivity, Specificity, Predictive Values and Likelihood Ratios," in *StatPearls* [Internet]. Treasure Island, FL, USA: StatPearls Publishing, 2025 Jan–. Updated Mar. 6, 2023. Available: https://www.ncbi.nlm.nih.gov/books/NBK557491/

[5] M. F. Kurniawan and D. A. Megawaty, "Comparison of Logistic Regression, Random Forest, Support Vector Machine (SVM) and K-Nearest Neighbor (KNN) Algorithms in Diabetes Prediction," *Journal of Applied Informatics and Computing*, vol. 9, no. 5, Oct. 2025, doi: 10.30871/jaic.v9i5.9815.

[6] B. Siswoyo and M. I. Nurhafidz, "Penerapan Algoritma Random Forest Untuk Prediksi Risiko Diabetes Berdasarkan Data Kesehatan Pasien," *Jurnal Teknologi Informasi Digital*, vol. 1, no. 1, 2025.

[7] R. Guido, S. Ferrisi, D. Lofaro, and D. Conforti, "An overview on the advancements of support vector machine models in healthcare applications: a review," Information, vol. 15, no. 4, p. 235, Apr. 2024, doi: 10.3390/info15040235.

[8] I. U. Rehman, M. M. Nasralla, and N. Y. Philip, "Multilayer perceptron neural network-based QoS-aware, content-aware and device-aware QoE prediction model: A proposed prediction model for medical ultrasound streaming over small cell networks," Electronics, vol. 8, no. 2, p. 194, Feb. 2019, doi: 10.3390/electronics8020194.


---

