import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("/content/Iris.csv")

print("Dataset Loaded Successfully!")

# Select numerical features
X = df[['SepalLengthCm', 'SepalWidthCm',
        'PetalLengthCm', 'PetalWidthCm']]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 11), wcss, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Cluster sum of squares")

plt.grid()
plt.show()


# Apply K-Means with Optimal K = 3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

df['Cluster'] = kmeans.fit_predict(X_scaled)

# Display Results
print("\nClustering Results:")
print(df[['SepalLengthCm', 'SepalWidthCm',
          'PetalLengthCm', 'PetalWidthCm', 'Cluster']])

# Cluster Distribution
print("\nNumber of Samples in Each Cluster:")
print(df['Cluster'].value_counts())
