# Creating movie recomendation system using tensorflow, k-means, clustering and Neurel network it is a sample work where data are extracted thorugh a link
# Remove '#' symbol from line 5,6,7 for smooth runing of code


# pip install pandas
# pip install -U scikit-learn
# pip install tensorflow

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow as tf



# 1.Collecting and processing data:
# Read movie data from a URL
url = 'https://www.imdb.com/search/title/?groups=top_250&sort=user_rating'
movies_df = pd.read_csv(url)

# Preprocess the data by removing missing or irrelevant information
movies_df = movies_df.dropna()
movies_df = movies_df[movies_df['year'] >= 1990]



# 2.Using K-means clustering to group similar movies together:
# Define the number of clusters
n_clusters = 5

# Extract features for the movie data
movie_features = movies_df[['imdb_score', 'duration', 'genra', 'year', 'movie_name']]

# Standardize the features
scaler = StandardScaler()
movie_features_scaled = scaler.fit_transform(movie_features)

# Apply K-means clustering
kmeans = KMeans(n_clusters=n_clusters)
cluster_assignments = kmeans.fit_predict(movie_features_scaled)



# 3.Building and training a TensorFlow model:
# Define the neural network architecture
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10, activation='relu', input_shape=(3,)))
model.add(tf.keras.layers.Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model on the movie data
model.fit(movie_features_scaled, cluster_assignments, epochs=10, batch_size=32)



# 5.Saving data to a CSV file
movies_df.to_csv("moviecon.csv", index=False)