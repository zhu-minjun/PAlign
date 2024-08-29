import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np


data_size = 300

df = pd.read_csv('../PAPI/IPIP_NEO_120.csv') # download from https://drive.google.com/file/d/1KRhpTCwSMS47GYnmHwYRPnmxF6FOGYTf/view?usp=sharing
features = df.columns[df.columns.str.startswith('i')]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=300, random_state=42)
kmeans.fit(X_scaled)

centers = kmeans.cluster_centers_

closest_data_index = []
for center in centers:
    distances = np.linalg.norm(X_scaled - center, axis=1)
    closest_index = np.argmin(distances)
    while closest_index in closest_data_index:
        distances[closest_index] = np.inf
        closest_index = np.argmin(distances)
    closest_data_index.append(closest_index)

selected_samples = df.iloc[closest_data_index]

print("Selected samples:")
print(selected_samples)

selected_samples.to_json('IPIP/selected_IPIP300_samples.json', orient='records', indent=4)


file = pd.read_excel('IPIP/IPIP-NEO-ItemKey.xls')

train_index = file['Full#'].to_list()[:120]
test_index = file['Full#'].to_list()[120:]

text_file = []
for i in range(len(file)):
    item = file.iloc[i].to_list()
    text_file.append({'label_raw':item[4],'text':item[5],'label_ocean':item[3][0],'key':{'+':1,'-':-1}[item[2][0]]})

text_file = pd.DataFrame(text_file)
text_file.to_csv('mpi_300.csv')

with open('IPIP/mpi_300_split.json','w',encoding='utf-8') as f:
    json.dump({'train_index':train_index,'test_index':test_index},f)
