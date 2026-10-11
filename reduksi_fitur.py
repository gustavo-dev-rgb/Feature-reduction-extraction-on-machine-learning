import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# 1. Baca data
df = 'PRSA_Data_20130301-20170228'
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



