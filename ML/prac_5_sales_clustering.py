import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("sales_data_sample.csv", encoding="ISO-8859-1")
print(df.head())
print(df.shape)

data = df[['QUANTITYORDERED', 'PRICEEACH', 'SALES', 'MSRP']].dropna()

scaler = StandardScaler()
X = scaler.fit_transform(data)

wcss = []
for k in range(1, 11):
  km = KMeans(n_clusters=k, random_state=42, n_init=10)
  km.fit(X)
  wcss.append(km.inertia_)
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data = data.copy()
data['KMeans_Cluster'] = kmeans.fit_predict(X)
print("\nK-Means Cluster Counts:")
print(data['KMeans_Cluster'].value_counts())

sns.scatterplot(
    x=data['SALES'], y=data['QUANTITYORDERED'],
    hue=data['KMeans_Cluster'], palette='Set1'
)
plt.title("K-Means Clustering")
plt.show()

linked = linkage(X, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linked, truncate_mode='lastp', p=20)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points / Clusters")
plt.ylabel("Distance")
plt.show()

hierarchical = AgglomerativeClustering(n_clusters=3)
data['Hierarchical_Cluster'] = hierarchical.fit_predict(X)
print("\nHierarchical Cluster Counts:")
print(data['Hierarchical_Cluster'].value_counts())

print("\nFinal Clustered Data:")
print(data.head())
