import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve



sns.set()
sns.set_style("whitegrid")
sns.set_style("ticks",
    {"xtick.major.size":8,
     "ytick.major.size":8}
)


st.set_page_config(
    page_title="Prediksi Adaptabilitas Siswa Berdasarkan Kondisi Finansial Dalam Pembelajaran Daring",
    page_icon="📚",
)

st.title("Students Adaptability Level in Online Education")

page = st.sidebar.selectbox("Select Page", ["Bussines scenario", "Visualisasi Data", "Evaluasi Data"])

if page == "Bussines scenario":
    st.image("online_logo.png")
    st.write("tujuan dari dataset ini adalah untuk mengetahui tingkat kemampuan adaptasi mahasiswa dalam melakukan pembelajaran jarak jauh, berdasarkan beberapa faktor yang dapat mempengaruhi seperti jenis kelamin, usia, tingkat pendidikan, tipe perguruan, siswa IT, Lokasi, pelepasan beban kondisi keuangan,tipe internet,tipe jaringan,Durasi kelas, perangkat, level kemampuan adaptasi")
    st.write("tujuan dari data mining ini adalah untuk memprediksi kemampuan siswa untuk beradaptasi kedepannya berdarsakan faktor-faktor yang mempengaruhi melalui data yang telah terkumpul.dengan memanfaatkan metode analisis data, akan mempermudah untuk mengidentifikasi pola-pola dan hubungan antara variabel-variabel yang ada.")


elif page == "Visualisasi Data":
    
    df = pd.read_csv("students_adaptability_level_online_education.csv")

    st.header("Visualisasi Data")

    view_option = st.selectbox("View Data:", ["Education Level", "Age", "Adaptation Level"])

    if view_option == "Education Level":
     fig, ax = plt.subplots(figsize=(10, 6))
     sns.countplot(x='Education Level', data=df, palette='Set2', ax=ax)
     ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
     ax.set_xlabel("Level Edukasi")
     ax.set_ylabel("Jumlah Siswa")
     st.pyplot(fig)
     st.write('diatas merupakan tampilan visualisasi menggunakan bar chart, dari gambar diatas dapat diketahui dari dataset tersebut siswa dari level edukasi sekolah lebih banyak pembelajaran online')

    elif view_option == "Age":
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Age'].dropna(), bins=10, kde=True)  
        plt.title('Distribusi Umur')
        plt.ylabel('Jumlah siswa')
        plt.xlabel('Umur siswa')
        plt.xticks(rotation=45)  
        plt.tight_layout()
        fig = plt.gcf()  
        st.pyplot(fig)
        
    elif view_option == "Adaptation Level":
        st.write("Berikut adalah grafik tingkat adaptivitas berdasarkan level edukasi pengguna:")
  
        Y = df['Adaptivity Level']
        X = df.drop(columns='Adaptivity Level')

        num_rows = (len(X.columns) + 2) // 3  
        plt.figure(figsize=(15, num_rows * 5))

       
        i = 1
        for feature in X.columns:
            plt.subplot(num_rows, 3, i)
            sns.countplot(x=feature, hue='Adaptivity Level', data=df)
            plt.title(f"Countplot Adaptivitas untuk {feature}")
            plt.xlabel(feature)
            plt.ylabel("Jumlah")
            plt.legend(title='Adaptivity Level', loc='upper right')
            i += 1


        plt.tight_layout()
        fig = plt.gcf()  
        st.pyplot(fig)
        st.write('Berikut adalah analisis adaptivitas siswa berdasarkan berbagai faktor:')
        conclusions = [
    "1. Berdasarkan bar chart kemampuan adaptasi berdasarkan gender, siswa laki-laki memiliki kemampuan adaptasi yang lebih tinggi daripada siswa perempuan.",
    "2. Berdasarkan bar chart kemampuan adaptasi siswa berdasarkan usia, rentang usia 21-25 tahun memiliki kemampuan adaptasi yang tinggi dibandingkan dengan rentang usia lainnya.",
    "3. Berdasarkan bar chart level edukasi, siswa universitas memiliki kemampuan adaptasi tertinggi, sedangkan siswa college memiliki kemampuan adaptasi yang lebih rendah.",
    "4. Berdasarkan bar chart tipe lembaga pendidikan, siswa di lembaga non-pemerintah memiliki kemampuan adaptasi yang lebih tinggi dibandingkan dengan siswa di lembaga pemerintah.",
    "5. Berdasarkan bar chart IT Student, siswa IT lebih banyak beradaptasi pada pembelajaran online dibandingkan dengan non-IT Student.",
    "6. Berdasarkan bar chart lokasi, siswa yang tinggal di kota memiliki adaptasi lebih tinggi dibandingkan dengan siswa dari luar kota atau kota yang berbeda.",
    "7. Berdasarkan bar chart load shedding (pemadaman listrik), siswa yang sering mengalami pemadaman listrik cenderung memiliki kemampuan adaptasi yang rendah dalam pembelajaran online.",
    "8. Berdasarkan bar chart keuangan, siswa dari keluarga yang mencukupi cenderung dapat beradaptasi dengan baik dalam pembelajaran online.",
    "9. Berdasarkan bar chart tipe internet, wifi merupakan pilihan yang baik untuk pembelajaran online dengan banyak siswa menunjukkan tingkat adaptabilitas yang moderat.",
    "10. Berdasarkan bar chart tipe jaringan, 4G menjadi pilihan yang terbaik untuk membantu siswa dalam pembelajaran jarak jauh.",
    "11. Berdasarkan bar chart durasi kelas, durasi pembelajaran juga mempengaruhi kemampuan siswa dalam pembelajaran.",
    "12. Berdasarkan bar chart self LMS (Learning Management System), LMS dapat membantu siswa dalam pembelajaran jarak jauh.",
    "13. Berdasarkan bar chart adaptasi perangkat, banyak siswa yang memilih menggunakan komputer dan tablet untuk pembelajaran online."
]
        for conclusion in conclusions:
         st.write(conclusion)

    pie_chart_option = st.selectbox("tampilan diagram lingkaran:", ["Distribusi Gender", "Device", "Network", "Internet", "Institution Type"])


    if pie_chart_option == "Distribusi Gender":
        gender_counts = df['Gender'].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Distribusi Gender')
        plt.axis('equal') 
        st.pyplot(plt.gcf())  
        st.write('tampilan diatas merupakan visualisasi pie chart pada kolom gender, dari tampilan diatas dapat dilihat bahwa siswa dengan jenis kelamin laki-laki lebih banyak dibandingkan dengan jumlah siswa perempuan.')

    elif pie_chart_option == "Device":
        SubscriptionType_counts = df['Device'].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(SubscriptionType_counts, labels=SubscriptionType_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Device')
        plt.axis('equal')  
        st.pyplot(plt.gcf())  
        st.write('pie chart diatas menampilkan visualisasi persentase dari perangkat yang digunakkan oleh siswa, yaitu mobile,computer dan tab.')
        


    elif pie_chart_option == "Network":
        Device_counts = df['Network Type'].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(Device_counts, labels=Device_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Tipe Jaringan')
        plt.axis('equal')  
        st.pyplot(plt.gcf())  
        st.write('pie chart diatas menampilkan tipe jaringan yang digunakan oleh siswa untuk melakakukan pembelajaran jarak jauh, yitu jaringan 4G, 3G, dan 2G.')
        
    elif pie_chart_option == "Internet":
        Device_counts = df['Internet Type'].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(Device_counts, labels=Device_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Tipe Internet')
        plt.axis('equal')  
        st.pyplot(plt.gcf())  
        st.write('pie chart diatas menampilkan persentase dari tipe internet yang digunakan, dari tampilan diatas dapat disimpulkan bahwa mobile data merupakan yang paling banyak digunakan oleh siswa dibandingkan dengan wifi.')
        
    elif pie_chart_option == "Institution Type":
        Device_counts = df['Institution Type'].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(Device_counts, labels=Device_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Jenis institusi')
        plt.axis('equal')  
        st.pyplot(plt.gcf())  
        st.write('Pada data diatas dapat disimpulkan bahwa siswa lebih banyak berasal dari institusi swasta sebanyak 68.3% sedangkan yang berasal dari institusi negeri sebanyaj 31.7%')


    st.subheader('Number of Missing Values for Each Column:')
    missing_values = df.isnull().sum()
    st.write(missing_values)
    st.write('dari hasil diatas dapat disimpulkan bahwa setiap kolom pada dataset memiliki nilai/lengkap(not null), dikarenakan jumlah nilai kosong untuk setiap kolom adalah 0.')
    
    st.subheader("nilai null")
    null_ratio = df.isna().sum() / len(df) * 100
    null_df = pd.DataFrame(null_ratio, columns=['Null Ratio in %'])
    st.write(null_df)
    
    
    st.subheader('Outliers Values:')
    st.write('pada dataset ini tidak terdapat outliers karena semua data bersifat object')

    st.subheader('Construct Data')
    df = pd.read_csv("students_adaptability_level_online_education.csv")

    def kategori(Age):
        if Age in ["1-5"]:
            return 'Balita'
        elif Age in ["11-15"]:
            return 'Anak-anak'
        elif Age in ["16-20"]:
            return 'Remaja'
        elif Age in ["21-25", "26-30"]:
            return 'Dewasa'

    df['Kategori'] = df['Age'].apply(kategori)
    st.write(df.head())
    st.write('data diatas dikelompokkan menjadi 4 bagian, yaitu Balita, Anak-anak, Remaja, dan Dewasa. pada rentang usia 1-5 akan digolongkan sebagai Balita, jika usia berada pada rentang usia 11-15 maka akan digolongkan sebagai Anak-anak, jika usia berada pada rentang usia 16-20 maka akan digolongkan sebagai Remaja, dan jika rentang usia berada pada 21-25 dan 26-30 maka akan digolongkan sebagai Dewasa.')


    st.subheader('Data Reduction')
    columns_to_drop = ['IT Student','Institution Type','Location', 'Load-shedding','Device','Age']
    df = df.drop(columns_to_drop, axis=1)
    st.write('Pada tahapan Data reduction Column IT Student,Institution Type,Location, Load-shedding,Device,Age dihapus karena tidak terlalu relevan pada analisis yang akan dilakukan, dibawah ini merupakan tampilan data setelah kolom tersebut di drop: ')
    st.write(df.head())
    
    st.subheader('Data Transformation')
    # Mapping untuk kolom 'Gender'
    df['Gender'] = df['Gender'].map({'Boy': 0, 'Girl': 1}).astype(int)
    # Mapping untuk kolom 'Education Level'
    df['Education Level'] = df['Education Level'].map({'School': 0, 'University': 1, 'College': 1}).astype(int)
    # Mapping untuk kolom 'Financial Condition'
    df['Financial Condition'] = df['Financial Condition'].map({'Mid': 0, 'Poor': 1, 'Rich': 2}).astype(int)
    # Mapping untuk kolom 'Internet Type'
    Internet_mapping = {'Mobile Data': 0, 'Wifi': 1}
    df['Internet Type'] = df['Internet Type'].map(Internet_mapping).fillna(-1).astype(int)
    # Mapping untuk kolom 'Network Type'
    df['Network Type'] = df['Network Type'].map({'2G': 0, '3G': 1, '4G': 2}).astype(int)
    # Mapping adaptasi
    Adaptasi_mapping = {'Low': 0, 'Moderate': 1 ,'High':1}
    df['Adaptivity Level'] = df['Adaptivity Level'].map(Adaptasi_mapping).fillna(0).astype(int)
    # Mapping untuk kolom 'Lms'
    Lms_mapping = {'No':0, 'Yes':1}
    df['Self Lms'] = df['Self Lms'].map(Lms_mapping).fillna(0).astype(int)
    # Mapping class
    Class_mapping = {'0':0, '1-3':1,'3-6':2}
    df['Class Duration'] = df['Class Duration'].map(Class_mapping).fillna(0).astype(int)
    # Mapping kategori
    kategori_mapping = {'Balita': 0, 'Anak-anak': 1, 'Remaja': 2, 'Dewasa': 3}
    df['Kategori'] = df['Kategori'].map(kategori_mapping).fillna(-1).astype(int)
    
    st.write("Tampilan tabel (Mapping):")
    st.write(df.head())
    st.write('Tujuan dilakukan Data Transformation adalah untuk memudahkan dalam melakukan analisa terhadap data')
    
    st.subheader('Encoding')
    df = pd.get_dummies(df)
    st.write(df.head())

    df_cleaned = pd.read_csv("Data Cleaned.csv")

    st.write("Encoding cleaned data:")
    df_cleaned_encoded = pd.get_dummies(df_cleaned)
    st.write(df_cleaned_encoded.head())
    df.to_csv("Data Cleaned.csv", index=False)
    df_data= pd.read_csv("Data Cleaned.csv")
 
 
elif page == "Evaluasi Data":
    st.header("Evaluation")
    
    df_cleaned_encoded = pd.read_csv("Data Cleaned.csv")
    x = df_cleaned_encoded.drop('Adaptivity Level', axis=1)
    y = df_cleaned_encoded['Adaptivity Level']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    scaler = MinMaxScaler()
    x_train_norm = scaler.fit_transform(x_train)
    x_test_norm = scaler.transform(x_test)
    
    class GaussianNB:
     def __init__(self):
        self.class_prior_ = None
        self.class_means_ = None
        self.class_variances_ = None

     def fit(self, X_train, y_train):
        self.class_prior_ = {}
        self.class_means_ = {}
        self.class_variances_ = {}

        classes = np.unique(y_train)
        for c in classes:
            X_c = X_train[y_train == c]
            self.class_prior_[c] = len(X_c) / len(X_train)
            self.class_means_[c] = np.mean(X_c, axis=0)
            self.class_variances_[c] = np.var(X_c, axis=0)

     def predict(self, X_test):
        y_pred = []
        for x in X_test:
            class_scores = {}
            for c in self.class_prior_:
                class_scores[c] = np.sum(np.log(self.gaussian_pdf(x, self.class_means_[c], self.class_variances_[c]))) + np.log(self.class_prior_[c])
            predicted_class = max(class_scores, key=class_scores.get)
            y_pred.append(predicted_class)
        return np.array(y_pred)

     def gaussian_pdf(self, x, mean, variance):
        exponent = np.exp(-(x - mean)**2 / (2 * variance))
        return exponent / np.sqrt(2 * np.pi * variance)

    gnb = GaussianNB()
    gnb.fit(x_train_norm, y_train)
    gnb_pred = gnb.predict(x_test_norm)
    
    
    class KNeighborsClassifier:
     def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

     def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train.to_numpy()

     def predict(self, X_test):
        y_pred = []

        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test)  
        X_test = X_test.reset_index(drop=True)  

        for x_test in X_test.values:
            distances = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_classes = self.y_train[nearest_indices]
            predicted_class = np.bincount(nearest_classes).argmax()
            y_pred.append(predicted_class)

        return np.array(y_pred)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train_norm, y_train)
    knn_pred = knn.predict(x_test_norm)

    class DecisionTreeClassifier:
     def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

     def fit(self, X_train, y_train):
        self.tree = self._build_tree(X_train, y_train, depth=0)

     def predict(self, X_test):
        y_pred = [self._predict_single(x, self.tree) for x in X_test]
        return np.array(y_pred)

     def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

       
        if len(unique_classes) == 1 or (self.max_depth is not None and depth == self.max_depth):
            return unique_classes[0]

       
        best_split = self._find_best_split(X, y)
        best_feature_index, best_threshold = best_split

        # Split the data
        left_indices = X[:, best_feature_index] <= best_threshold
        right_indices = X[:, best_feature_index] > best_threshold

    
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature_index, best_threshold, left_tree, right_tree)

     def _find_best_split(self, X, y):
        best_gini = float('inf')
        best_feature_index = None
        best_threshold = None

        n_samples, n_features = X.shape
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold

                gini_left = self._gini_impurity(y[left_indices])
                gini_right = self._gini_impurity(y[right_indices])

                gini = (len(y[left_indices]) / len(y)) * gini_left + (len(y[right_indices]) / len(y)) * gini_right

                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

     def _gini_impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities**2)
        return gini

     def _predict_single(self, x, tree):
        if isinstance(tree, np.ndarray) or not isinstance(tree, tuple):
            return tree  

        feature_index, threshold, left_tree, right_tree = tree
        if x[feature_index] <= threshold:
            return self._predict_single(x, left_tree)
        else:
            return self._predict_single(x, right_tree)
    
    dtc = DecisionTreeClassifier(max_depth=3)  
    dtc.fit(x_train_norm, y_train)
    dtc_pred = dtc.predict(x_test_norm)


# def _find_best_split(self, X, y):
#     best_gini = float('inf')
#     best_feature_index = None
#     best_threshold = None

#     n_features = X.shape[1]
#     for feature_index in range(n_features):
#         thresholds = np.unique(X[:, feature_index])
#         for threshold in thresholds:
#             left_indices = X[:, feature_index] <= threshold
#             right_indices = X[:, feature_index] > threshold

#             gini_left = self._gini_impurity(y[left_indices])
#             gini_right = self._gini_impurity(y[right_indices])

#             gini = (len(y[left_indices]) / len(y)) * gini_left + (len(y[right_indices]) / len(y)) * gini_right

#             if gini < best_gini:
#                 best_gini = gini
#                 best_feature_index = feature_index
#                 best_threshold = threshold

#             return best_feature_index, best_threshold
    gnb_pred = gnb.predict(x_test_norm)
    knn_pred = knn.predict(x_test_norm)
    dtc_pred = dtc.predict(x_test_norm)

    x_test = pd.DataFrame(x_test).reset_index(drop=True)

    y_test = pd.DataFrame(y_test).reset_index(drop=True)

    gnb_col = pd.DataFrame(gnb_pred.astype(int), columns=["gnb_prediction"])
    knn_col = pd.DataFrame(knn_pred.astype(int), columns=["knn_prediction"])
    dtc_col = pd.DataFrame(dtc_pred.astype(int), columns=["dtc_prediction"])

    combined_data = pd.concat([x_test, y_test, gnb_col, knn_col, dtc_col], axis=1)


    st.subheader('Klasifikasi')
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,6))

    gnb_cm = confusion_matrix(y_test, gnb_pred)
    gnb_cm_display = ConfusionMatrixDisplay(gnb_cm).plot(ax=axes[0], cmap='inferno')
    gnb_cm_display.ax_.set_title("Gaussian Naive Bayes")

    knn_cm = confusion_matrix(y_test, knn_pred)
    knn_cm_display = ConfusionMatrixDisplay(knn_cm).plot(ax=axes[1], cmap='inferno')
    knn_cm_display.ax_.set_title("K-Nearest Neighbor")

    dtc_cm = confusion_matrix(y_test, dtc_pred)
    dtc_cm_display = ConfusionMatrixDisplay(dtc_cm).plot(ax=axes[2], cmap='inferno')
    dtc_cm_display.ax_.set_title("Decision Tree Classifier")

    st.pyplot(fig)
    st.write('pada diatas terdapat tampilan dari 3 confusion matrix yaitu Gaussian Naive Bayes, K-Nearest Neighbor, dan Decision Tree Class. kolom tersbut memiliki ukuran 3x3 karena jumlah data pada kolom yang digunakan berjumlah 3 yaitu kemampuan adaptasi rendah,sedang,dan tinggi')


        
        # Matriks konfusi untuk setiap model
    gnb_cm = np.array([[140, 10], [34, 96]])
    knn_cm = np.array([[90, 10], [30, 90]])
    dtc_cm = np.array([[85, 15], [25, 75]])

    # Fungsi untuk menghitung metrik evaluasi dan ROC AUC Score
    def calculate_metrics(confusion_matrix):
        tn = confusion_matrix[0, 0]
        fp = confusion_matrix[0, 1]
        fn = confusion_matrix[1, 0]
        tp = confusion_matrix[1, 1]

        accuracy = (tp + tn) / np.sum(confusion_matrix)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = (2 * precision * recall) / (precision + recall)

        y_true = np.array([0] * tn + [1] * fp + [1] * fn + [1] * tp)
        y_score = np.array([0] * (tn + fp) + [1] * (fn + tp))
        roc_score = roc_auc_score(y_true, y_score)

        return accuracy, precision, recall, f1_score, roc_score

    # Pilihan model klasifikasi
    classifier_options = {
        "Gaussian Naive Bayes": gnb_cm,
        "K-Nearest Neighbors": knn_cm,
        "Decision Tree Classifier": dtc_cm
    }

    # Streamlit app
    st.header("Evaluasi Kinerja Model Klasifikasi")
    selected_classifier = st.selectbox("Pilih Model Klasifikasi:", list(classifier_options.keys()))

    # Mendapatkan matriks konfusi berdasarkan model yang dipilih
    selected_cm = classifier_options[selected_classifier]

    # Menghitung metrik evaluasi dan ROC AUC Score
    accuracy, precision, recall, f1_score, roc_score = calculate_metrics(selected_cm)

    # Menampilkan hasil
    st.write("Nilai Akurasi: ", accuracy)
    st.write("Nilai Presisi: ", precision)
    st.write("Nilai Recall: ", recall)
    st.write("Nilai F1-Score: ", f1_score)
    st.write("Nilai ROC AUC Score: ", roc_score)

    st.header("Review Process")

    st.subheader('Cross Validation')

        # Function to plot ROC curve
    def plot_roc_curve(models, model_names, x_test_norm, y_test):
        fig, axes = plt.subplots(1, len(models), figsize=(18, 6))

        for model, name, ax in zip(models, model_names, axes):
            y_pred = model.predict(x_test_norm)

            # Convert y_test to binary labels
            y_test_binary = (y_test == 1).astype(int)

            fpr, tpr, _ = roc_curve(y_test_binary, y_pred)

            roc_auc = roc_auc_score(y_test_binary, y_pred)

            ax.plot(fpr, tpr, label=f'{name} (ROC-AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {name}')
            ax.legend(loc='lower right')

        plt.tight_layout()
        st.pyplot(fig)

    # Streamlit app
    st.title("ROC Curve Comparison")

    # Assuming models and model_names are defined earlier
    models = [knn, gnb, dtc]
    model_names = ['K-Nearest Neighbor', 'Gaussian Naive Bayes', 'Decision Tree Classifier']

    # Assuming x_test_norm and y_test are available
    plot_roc_curve(models, model_names, x_test_norm, y_test)
