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

