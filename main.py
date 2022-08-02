import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Denotes how many 90s the player has needed to play to be considered by the KMeans
MINIMUM_GAMES_PLAYED = 6

# Load the data from the csv file
df = pd.read_csv("stats.csv")

# Clean the data to ensure that  the players have played more than the minimum 90s
df = df[df['90s'] >= MINIMUM_GAMES_PLAYED].reset_index()

# Drop the columns that are not needed for the kmeans algorithm
# Age,Born,MP,Starts,Min,90s
kmeans_df = df
kmeans_df = kmeans_df.drop('Rk', axis=1)
kmeans_df = kmeans_df.drop('Player', axis=1)
kmeans_df = kmeans_df.drop('Nation', axis=1)
kmeans_df = kmeans_df.drop('Pos', axis=1)
kmeans_df = kmeans_df.drop('Squad', axis=1)
kmeans_df = kmeans_df.drop('Comp', axis=1)
kmeans_df = kmeans_df.drop('Age', axis=1)
kmeans_df = kmeans_df.drop('Born', axis=1)
kmeans_df = kmeans_df.drop('MP', axis=1)
kmeans_df = kmeans_df.drop('Starts', axis=1)
kmeans_df = kmeans_df.drop('Min', axis=1)
kmeans_df = kmeans_df.drop('90s', axis=1)

columns = list(kmeans_df)

scaler = StandardScaler()
kmeans_df = scaler.fit_transform(kmeans_df.to_numpy())
kmeans_df = pd.DataFrame(kmeans_df, columns=columns)

ss = []
for n in range(2,10):
    km = KMeans(n_clusters=n)
    km.fit(kmeans_df)
    label = km.predict(kmeans_df)
    silhouette = silhouette_score(kmeans_df, label)
    ss.append(silhouette)

plt.plot(range(2,10), ss)

km = KMeans(n_clusters=4)
y = km.fit_predict(kmeans_df)

kmeans_df['Cluster'] = y

# We want 2 principal components for the graph
pri_comp_anal = PCA(n_components=2)

# Performs PCA on the data
principal_components = pri_comp_anal.fit_transform(kmeans_df)

pca_df = pd.DataFrame(principal_components, columns=['Principal Component 1', "Principal Component 2"])

kmeans_df = pd.concat([kmeans_df, pca_df], axis=1)

km = KMeans(n_clusters=4)
y = km.fit_predict(kmeans_df)

kmeans_df['Cluster'] = y

# We want 2 principal components for the graph
pri_comp_anal = PCA(n_components=2)

# Performs PCA on the data
principal_components = pri_comp_anal.fit_transform(kmeans_df)

pca_df = pd.DataFrame(principal_components, columns=['Principal Component 1', "Principal Component 2"])

kmeans_df = pd.concat([kmeans_df, pca_df], axis=1)

kmeans_df['Player'] = df['Player']
kmeans_df['Comp'] = df['Comp']
print(kmeans_df)
prem = kmeans_df[kmeans_df['Comp'] == 'eng Premier League']
df_one = kmeans_df[kmeans_df['Cluster'] == 0]
df_two = kmeans_df[kmeans_df['Cluster'] == 1]
df_three = kmeans_df[kmeans_df['Cluster'] == 2]
df_four = kmeans_df[kmeans_df['Cluster'] == 3]



plt.show()

plt.title("KMeans clustering for top 5 league goalkeepers")
plt.scatter(df_one['Principal Component 1'], df_one['Principal Component 2'], label="Cluster One")
plt.scatter(df_two['Principal Component 1'], df_two['Principal Component 2'], label="Cluster Two")
plt.scatter(df_three['Principal Component 1'], df_three['Principal Component 2'], label="Cluster Three")
plt.scatter(df_four['Principal Component 1'], df_four['Principal Component 2'], label="Cluster Four")

for x, y, name in zip(prem['Principal Component 1'], prem['Principal Component 2'], prem['Player']):
    print(str(x) + str(y) + name)
    plt.text(x, y, name,fontsize=5)

plt.legend()
plt.show()
