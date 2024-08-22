import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud

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

# Function to get the decade from the release year
def get_decade(year):
    return int(year) // 10 * 10

# Sidebar for Dropdowns
st.sidebar.title("Filter Options")
selected_genre = st.sidebar.selectbox("Select Genre", list(genre_to_subgenre.keys()))
selected_subgenre = st.sidebar.selectbox("Select Subgenre", genre_to_subgenre[selected_genre])

# Decade slider
min_year = int(data['track_album_release_date'].min()[:4])
max_year = int(data['track_album_release_date'].max()[:4])
decade_range = range(get_decade(min_year), get_decade(max_year) + 10, 10)
selected_decades = st.sidebar.slider("Select Decade(s)", min_value=min(decade_range), 
                                     max_value=max(decade_range), value=(min(decade_range), max(decade_range)), step=10)

# Filter data based on selections
data['decade'] = data['track_album_release_date'].apply(lambda x: get_decade(x[:4]))
filtered_data = data[(data['playlist_genre'] == selected_genre) & 
                     (data['playlist_subgenre'] == selected_subgenre) &
                     (data['decade'] >= selected_decades[0]) & 
                     (data['decade'] <= selected_decades[1])]

# Feature selection for top 10 display
feature_options = ['danceability', 'energy', 'loudness', 'speechiness', 
                   'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
selected_feature = st.sidebar.selectbox("Select Feature for Top 10", feature_options)

# Main Dashboard Title
st.title("Song Popularity Analysis Dashboard")

# Visualization 1: Word Cloud for Album Names
st.subheader("Word Cloud for Album Names")
album_names = ' '.join(filtered_data['track_album_name'].dropna().tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(album_names)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

# Visualization 2: Top 10 Artists/Albums for Selected Genre/Subgenre
st.subheader(f"Top 10 Artists in {selected_subgenre}")
top_artists_genre = filtered_data.groupby('track_artist')['track_popularity'].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
sns.barplot(x=top_artists_genre.values, y=top_artists_genre.index, ax=ax, orient='h')
ax.set_title(f"Top 10 Artists in {selected_subgenre}")
st.pyplot(fig)

st.subheader(f"Top 10 Albums in {selected_subgenre}")
top_albums_genre = filtered_data.groupby('track_album_name')['track_popularity'].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
sns.barplot(x=top_albums_genre.values, y=top_albums_genre.index, ax=ax, orient='h')
ax.set_title(f"Top 10 Albums in {selected_subgenre}")
st.pyplot(fig)

# Visualization 3: Radar Chart for Song Features
st.subheader("Radar Chart for Song Features")
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
ax.set_title(f"Radar Chart for Song Features in {selected_subgenre}")
st.pyplot(fig)

# Visualization 4: Top 10 Artists/Albums for Selected Feature
st.subheader(f"Top 10 Artists by {selected_feature.capitalize()}")
top_artists_feature = filtered_data.groupby('track_artist')[selected_feature].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
sns.barplot(x=top_artists_feature.values, y=top_artists_feature.index, ax=ax, orient='h')
ax.set_title(f"Top 10 Artists by {selected_feature.capitalize()} in {selected_subgenre}")
st.pyplot(fig)

st.subheader(f"Top 10 Albums by {selected_feature.capitalize()}")
top_albums_feature = filtered_data.groupby('track_album_name')[selected_feature].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
sns.barplot(x=top_albums_feature.values, y=top_albums_feature.index, ax=ax, orient='h')
ax.set_title(f"Top 10 Albums by {selected_feature.capitalize()} in {selected_subgenre}")
st.pyplot(fig)
