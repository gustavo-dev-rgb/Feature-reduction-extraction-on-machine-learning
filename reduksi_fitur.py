import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# 1. Baca data
df = r"D:\Machine Learning\feature reduction & extraction\Feature reduction & extraction on machine learning\PRSA_Data_20130301-20170228"
files = glob.glob(os.path.join(df, '*.csv'))

df = pd.concat(
    [pd.read_csv(file, na_values=['NA']) for file in files],
    ignore_index=True
)

print("Jumlah file:", len(files))
print("Ukuran dataset:", df.shape)
print("\nMissing value per kolom:")
print(df.isnull().sum())


#2. Data Cleaning(missing values, duplikasi, dll)
print(df.isnull().sum())
print("Duplikat:", df.duplicated().sum())
df = df.drop_duplicates()

fitur = [
    'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
    'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM'
]

data = df[fitur].copy()
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(data.median())
print("Sisa missing value:", data.isnull().sum().sum())

# 3. Normalisasi menggunakan minmax
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
print(data_scaled[:5])

# 4. Lakukan PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
print("Variansi yang dijelaskan:")
print(pca.explained_variance_ratio_)
print(
    "Total variansi:",
    pca.explained_variance_ratio_.sum()
)

# # 5. Clustering (kmeans, agglo, dbscan) pilih salah satu
# from sklearn.cluster import KMeans

# kmeans = KMeans(
#     n_clusters=3,
#     random_state=42,
#     n_init=10
# )

# cluster = kmeans.fit_predict(data_pca)
# hasil = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
# hasil['Cluster'] = cluster

# print(hasil.head())
# print(hasil['Cluster'].value_counts())
# plt.figure(figsize=(10, 6))

# plt.scatter(
#     hasil['PC1'],
#     hasil['PC2'],
#     c=hasil['Cluster'],
#     cmap='viridis',
#     s=10
# )

# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('K-Means Clustering - Beijing Air Quality')
# plt.show()


