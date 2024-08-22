import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file
data = pd.read_csv('spotify_songs.csv')

# Mapping of genres to their corresponding subgenres
genre_to_subgenre = {
    'edm': ['big room', 'electro house', 'pop edm', 'progressive electro house'],
    'latin': ['latin hip hop', 'latin pop', 'reggaeton', 'tropical'],
    'pop': ['dance pop', 'electropop', 'indie poptimism', 'post-teen pop'],
    'r&b': ['neo soul', 'new jack swing', 'urban contemporary'],
    'rap': ['gangster rap', 'hip hop', 'hip pop', 'southern hip hop', 'trap'],
    'rock': ['album rock', 'classic rock', 'hard rock', 'permanent wave']
}

# Sidebar for Dropdowns
st.sidebar.title("Filter Options")
selected_genre = st.sidebar.selectbox("Select Genre", list(genre_to_subgenre.keys()))
selected_subgenre = st.sidebar.selectbox("Select Subgenre", genre_to_subgenre[selected_genre])

# Filter data based on selections
filtered_data = data[(data['playlist_genre'] == selected_genre) & 
                     (data['playlist_subgenre'] == selected_subgenre)]

# Feature selection for top 10 display
feature_options = ['danceability', 'energy', 'loudness', 'speechiness', 
                   'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
selected_feature = st.sidebar.selectbox("Select Feature for Top 10", feature_options)

# Main Dashboard Title
st.title("Song Popularity Analysis Dashboard")

# Visualization 1: Heatmaps for All Attributes
st.subheader("Heatmaps for All Attributes")
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.tight_layout(pad=3.0)
for i, feature in enumerate(feature_options):
    row, col = divmod(i, 3)
    sns.heatmap(filtered_data.pivot_table(index='track_artist', 
                                          columns='track_album_name', 
                                          values=feature, 
                                          aggfunc='mean'), 
                ax=axes[row, col], cmap="YlGnBu")
    axes[row, col].set_title(f'Heatmap of {feature} by Artist and Album')

st.pyplot(fig)

# Visualization 2: Improved Top 10 Artists/Albums Display (Horizontal Bar Charts)
st.subheader(f"Top 10 Artists by {selected_feature.capitalize()}")
top_artists = filtered_data.groupby('track_artist')[selected_feature].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
sns.barplot(x=top_artists.values, y=top_artists.index, ax=ax, orient='h')
ax.set_title(f"Top 10 Artists in {selected_subgenre} by {selected_feature.capitalize()}")
st.pyplot(fig)

st.subheader(f"Top 10 Albums by {selected_feature.capitalize()}")
top_albums = filtered_data.groupby('track_album_name')[selected_feature].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
sns.barplot(x=top_albums.values, y=top_albums.index, ax=ax, orient='h')
ax.set_title(f"Top 10 Albums in {selected_subgenre} by {selected_feature.capitalize()}")
st.pyplot(fig)

# Visualization 3: Radar Chart (with Normalization)
st.subheader("Radar Chart: Normalized Feature Averages")
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(filtered_data[feature_options])
normalized_avg_features = np.mean(normalized_features, axis=0)

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
labels = feature_options
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
stats = np.concatenate((normalized_avg_features, [normalized_avg_features[0]]))
angles += angles[:1]

ax.fill(angles, stats, color='blue', alpha=0.25)
ax.plot(angles, stats, color='blue', linewidth=2)
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_title(f"Normalized Feature Averages for {selected_subgenre}")
st.pyplot(fig)
