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

