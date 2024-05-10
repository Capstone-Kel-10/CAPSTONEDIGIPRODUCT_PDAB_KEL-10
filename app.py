import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# Set style for visualizations
df = pd.read_csv("students_adaptability_level_online_education.csv")
sns.set()
sns.set_style("whitegrid")
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

# Set Streamlit page configuration
st.set_page_config(
    page_title="Prediksi Adaptabilitas Siswa Berdasarkan Kondisi Finansial Dalam Pembelajaran Daring",
    page_icon="📚",
)

# Main title
st.title("Students Adaptability Level in Online Education")

# Sidebar selection
page = st.sidebar.selectbox("Select Page", ["Business Scenario", "Visualization"])

if page == "Business Scenario":
    st.image("online_logo.png")
    st.write("Tujuan dari dataset ini adalah untuk mengetahui tingkat kemampuan adaptasi mahasiswa dalam melakukan pembelajaran jarak jauh, berdasarkan beberapa faktor yang dapat mempengaruhi seperti jenis kelamin, usia, tingkat pendidikan, tipe perguruan, siswa IT, lokasi, pemadaman beban, kondisi keuangan, tipe internet, tipe jaringan, durasi kelas, perangkat, dan level kemampuan adaptasi.")
    st.header('Latar Belakang Tema')
    st.write('Latar belakang kelompok kami memilih untuk mengangkat tema Prediksi Adaptabilitas Siswa Berdasarkan Kondisi Finansial dalam Pembelajaran Daring adalah untuk menganalisis lebih dalam mengenai pengaruh faktor finansial yang dapat memengaruhi aksesibilitas siswa terhadap teknologi yang diperlukan untuk pembelajaran daring, seperti perangkat keras dan konektivitas internet yang digunakan. Selain itu, keterbatasan finansial dapat berdampak pada lingkungan belajar siswa di rumah, seperti ruang dan suasana yang mendukung pembelajaran. Dengan memahami hubungan antara kondisi finansial dan adaptabilitas siswa dalam pembelajaran daring, kita dapat mengidentifikasi tantangan utama yang dihadapi siswa dan mengembangkan strategi untuk meningkatkan kesetaraan akses dan kesempatan belajar.')
    st.header('Permasalahan')
    st.write('Masalah utama yang dihadapi dalam konteks ini adalah kesenjangan aksesibilitas dan adaptabilitas siswa terhadap pembelajaran daring sebagai akibat dari kondisi finansial yang beragam di antara mereka. Beberapa siswa mungkin memiliki akses terbatas atau bahkan tidak memiliki akses sama sekali terhadap perangkat keras seperti laptop atau tablet yang diperlukan untuk mengikuti pembelajaran daring. Selain itu, konektivitas internet juga bisa menjadi kendala bagi siswa yang tinggal di daerah dengan jaringan internet yang tidak stabil atau tidak tersedia.')
    st.header('Solusi')
    st.write(' untuk mengatasi masalah ini kelompok kami membuat web khusus webinar dengan harga yang terjangkau, tujuan dari web webinar ini adalah untuk meningkatkan kesadaran tentang pentingnya pembelajaran daring dan bagaimana kondisi finansial dapat memengaruhi aksesibilitas siswa terhadap pendidikan. Kami juga ingin meningkatkan pengetahuan siswa, orang tua, dan pendidik tentang sumber daya yang tersedia dan strategi yang dapat mereka gunakan untuk mengatasi hambatan finansial. dalam web tersebut juga terdapat Sesi-sesi pelatihan yang akan membantu siswa dan orang tua mereka merasa lebih percaya diri dalam menghadapi teknologi dan memaksimalkan manfaatnya dalam proses pembelajaran.')
elif page == "Visualization":
    df = pd.read_csv("students_adaptability_level_online_education.csv")

    st.header("Visualization")

    view_option = st.selectbox("View Data:", ["Education Level", "Age", "Adaptation Level"])

    if view_option == "Education Level":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='Education Level', data=df, palette='Set2', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel("Level Edukasi")
        ax.set_ylabel("Jumlah Siswa")
        st.pyplot(fig)
        st.write('Di atas merupakan tampilan visualisasi menggunakan bar chart, dari gambar di atas dapat diketahui dari dataset tersebut siswa dari level edukasi sekolah lebih banyak pembelajaran online.')

    elif view_option == "Age":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Age'].dropna(), bins=10, kde=True, ax=ax)
        ax.set_title('Distribusi Umur')
        ax.set_ylabel('Jumlah siswa')
        ax.set_xlabel('Umur siswa')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
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

    
    pie_chart_option = st.selectbox("Tampilan Diagram Lingkaran:", ["Distribusi Gender", "Device", "Network", "Internet", "Institution Type"])

    if pie_chart_option == "Distribusi Gender":
        gender_counts = df['Gender'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
        ax.set_title('Distribusi Gender')
        ax.axis('equal') 
        st.pyplot(fig)
        st.write('tampilan diatas merupakan visualisasi pie chart pada kolom gender, dari tampilan diatas dapat dilihat bahwa siswa dengan jenis kelamin laki-laki lebih banyak dibandingkan dengan jumlah siswa perempuan.')

    elif pie_chart_option == "Device":
        subscription_counts = df['Device'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(subscription_counts, labels=subscription_counts.index, autopct='%1.1f%%', startangle=140)
        ax.set_title('Device')
        ax.axis('equal')  
        st.pyplot(fig)
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
        st.write('Pada data diatas dapat disimpulkan bahwa siswa lebih banyak berasal dari institusi swasta sebanyak 68.3% sedangkan yang berasal dari institusi negeri sebanyak 31.7%')


    #####**Split Data**
    columns_to_drop = ['IT Student','Institution Type','Location', 'Load-shedding','Device','Age']

    # Menghapus kolom-kolom dengan tipe data object
    df = df.drop(columns_to_drop, axis=1)

    # Tampilkan DataFrame setelah penghapusan kolom
    print("\nDataFrame Setelah Penghapusan Kolom:")
    print(df)

    model_df=df.copy()

    model_df['Financial Condition'] = model_df['Financial Condition'].map({'Mid': 0, 'Poor': 1, 'Rich': 2})

    model_df['Education Level'] = model_df['Education Level'].map({'School': 0, 'University': 1, 'College': 1})

    model_df['Internet Type'] = model_df['Internet Type'].map({'Mobile Data': 0, 'Wifi': 1})

    model_df['Self Lms'] = model_df['Self Lms'].map({'No':0, 'Yes':1})

    model_df['Gender'] = model_df['Gender'].map({'Boy': 0, 'Girl': 1})

    model_df['Network Type'] = model_df['Network Type'].map({'2G': 0, '3G': 1, '4G': 2})

    model_df['Class Duration'] = model_df['Class Duration'].map({'0':0, '1-3':1,'3-6':2})

    x = model_df.drop('Adaptivity Level', axis=1)
    y = model_df['Adaptivity Level']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


    y_train_encoded = y_train.map({'Low': 0, 'Moderate': 1 ,'High':2})
    y_test_encoded = y_test.map({'Low': 0, 'Moderate': 1 ,'High':2})
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
                
    numeric = x_train.select_dtypes(include=['int64', 'float64']).columns

    numeric_transform = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

    category = x_train.select_dtypes(include=['object']).columns

    category_transform = Pipeline(steps=[
            ('onehot', OneHotEncoder())
        ])


    preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transform, numeric),
                ('cat', category_transform, category)
            ]
        )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


    pipeline = Pipeline([
            ('pca', PCA(n_components=10)),
            ('clf', RandomForestClassifier())

        ])

    class  NaiveBayesClassifier:
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

    pipeline_gnb = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', NaiveBayesClassifier())])

    pipeline_gnb.fit(x_train, y_train_encoded)


    class KNeighborsClassifier:
            def __init__(self, n_neighbors=5):
                self.n_neighbors = n_neighbors
                self.X_train = None
                self.y_train = None

            def fit(self, X_train, y_train):
                # Menyimpan X_train dan y_train (mengonversi ke array NumPy jika perlu)
                self.X_train = np.array(X_train)
                if isinstance(y_train, pd.DataFrame):
                    self.y_train = y_train.values.flatten()  # Mengambil nilai array jika y_train adalah DataFrame
                else:
                    self.y_train = np.array(y_train)

            def predict(self, X_test):
                y_pred = []

                # Mengonversi X_test ke DataFrame jika diperlukan
                if isinstance(X_test, np.ndarray):
                    X_test = pd.DataFrame(X_test)

                # Mengatur ulang indeks DataFrame X_test
                X_test = X_test.reset_index(drop=True)

                for x_test in X_test.values:
                    # Menghitung jarak Euclidean antara x_test dengan setiap titik dalam X_train
                    distances = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))

                    # Mendapatkan indeks k terdekat
                    nearest_indices = np.argsort(distances)[:self.n_neighbors]

                    # Mendapatkan kelas dari k tetangga terdekat
                    nearest_classes = self.y_train[nearest_indices]

                    # Memprediksi kelas dengan menggunakan mayoritas voting
                    predicted_class = np.bincount(nearest_classes).argmax()

                    # Menyimpan prediksi
                    y_pred.append(predicted_class)

                return np.array(y_pred)

    pipeline_knn = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', KNeighborsClassifier())])

    pipeline_knn.fit(x_train, y_train_encoded)

    class DecisionTreeClassifier:
            def __init__(self, max_depth=None):
                self.max_depth = max_depth
                self.tree = None

            def fit(self, X_train, y_train):
                # Mengonversi X_train dan y_train menjadi array NumPy
                X_train = np.array(X_train)
                y_train = np.array(y_train)

                # Membangun pohon keputusan
                self.tree = self._build_tree(X_train, y_train, depth=0)

            def predict(self, X_test):
                # Memprediksi untuk setiap sampel dalam X_test
                y_pred = [self._predict_single(x, self.tree) for x in X_test]
                return np.array(y_pred)

            def _build_tree(self, X, y, depth):
                n_samples, n_features = X.shape
                unique_classes = np.unique(y)

                # Stop criteria: jika hanya satu kelas atau kedalaman maksimum tercapai
                if len(unique_classes) == 1 or (self.max_depth is not None and depth >= self.max_depth):
                    return self._get_majority_class(y)

                # Temukan pemisahan terbaik
                best_split = self._find_best_split(X, y)
                best_feature_index, best_threshold = best_split

                if best_feature_index is None or best_threshold is None:
                    # Tidak ada pemisahan yang valid, kembalikan kelas mayoritas
                    return self._get_majority_class(y)

                # Pisahkan data berdasarkan fitur terbaik dan threshold
                left_indices = X[:, best_feature_index] <= best_threshold
                right_indices = ~left_indices

                if np.all(left_indices) or np.all(~left_indices):
                    # Semua sampel berada di satu cabang, kembalikan kelas mayoritas
                    return self._get_majority_class(y)

                # Bangun pohon secara rekursif untuk cabang kiri dan kanan
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
                        right_indices = ~left_indices

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
                    return tree  # Node daun (kelas yang diprediksi)

                feature_index, threshold, left_tree, right_tree = tree
                if x[feature_index] <= threshold:
                    return self._predict_single(x, left_tree)
                else:
                    return self._predict_single(x, right_tree)

            def _get_majority_class(self, y):
                unique_classes, counts = np.unique(y, return_counts=True)
                majority_class = unique_classes[np.argmax(counts)]
                return majority_class

    pipeline_dtc = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', DecisionTreeClassifier())])

    pipeline_dtc.fit(x_train, y_train_encoded)

    gnb_pred = pipeline_gnb.predict(x_test)
    knn_pred = pipeline_knn.predict(x_test)
    dtc_pred = pipeline_dtc.predict(x_test)

    gnb_pred = pd.Series(gnb_pred).map({0: 'Low', 1: 'Moderate', 2: 'High'})
    knn_pred = pd.Series(knn_pred).map({0: 'Low', 1: 'Moderate', 2: 'High'})
    dtc_pred = pd.Series(dtc_pred).map({0: 'Low', 1: 'Moderate', 2: 'High'})

    def undummify(df, prefix_sep="_"):
            cols2collapse = {
                item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
            }
            series_list = []
            for col, needs_to_collapse in cols2collapse.items():
                if needs_to_collapse:
                    undummified = (
                        df.filter(like=col)
                        .idxmax(axis=1)
                        .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                        .rename(col)
                    )
                    series_list.append(undummified)
                else:
                    series_list.append(df[col])
            undummified_df = pd.concat(series_list, axis=1)
            return undummified_df

    x_test = undummify(x_test)

        # Mengonversi list menjadi numpy array
    gnb_pred_array = np.array(gnb_pred)
    knn_pred_array = np.array(knn_pred)
    dtc_pred_array = np.array(dtc_pred)


    gnb_col = pd.DataFrame(gnb_pred_array, columns=["gnb_prediction"])
    knn_col = pd.DataFrame(knn_pred_array, columns=["knn_prediction"])
    dtc_col = pd.DataFrame(dtc_pred_array, columns=["dtc_prediction"])

    # Reset index untuk x_test dan y_test
    x_test_df = pd.DataFrame(x_test).reset_index(drop=True)
    y_test_df = pd.DataFrame(y_test).reset_index(drop=True)

    # Menggabungkan data menjadi satu DataFrame
    combined_data = pd.concat([x_test_df, y_test_df, gnb_col, knn_col, dtc_col], axis=1)

    # Tampilkan data gabungan dalam aplikasi Streamlit
    st.header("Hasil Prediksi Model")
    st.write("Berikut adalah hasil prediksi dari model yang telah dilatih:")

    # Menampilkan DataFrame sebagai tabel di aplikasi Streamlit
    st.dataframe(combined_data)