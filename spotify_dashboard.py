import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
data = pd.read_csv('spotify_songs.csv')  # Ensure the CSV file is in the correct directory

# Example mapping of genres to their corresponding subgenres
genre_to_subgenre = {
    'edm': ['big room', 'electro house', 'pop edm', 'progressive electro house'],
    'latin': ['latin hip hop', 'latin pop', 'reggaeton', 'tropical'],
    'pop': ['dance pop', 'electropop', 'indie poptimism', 'post-teen pop'],
    'r&b': ['neo soul', 'new jack swing', 'urban contemporary'],
    'rap': ['gangster rap', 'hip hop', 'hip pop', 'southern hip hop', 'trap'],
    'rock': ['album rock', 'classic rock', 'hard rock', 'permanent wave']
}

st.title("Song Popularity Analysis Dashboard")

# Genre and Subgenre selection
selected_genre = st.selectbox("Select Genre", list(genre_to_subgenre.keys()))
selected_subgenre = st.selectbox("Select Subgenre", genre_to_subgenre[selected_genre])

# Filter data based on selections
filtered_data = data[(data['playlist_genre'] == selected_genre) & 
                     (data['playlist_subgenre'] == selected_subgenre)]

# Visualization 1: Scatter Plot (Tempo vs Danceability)
st.subheader("Scatter Plot: Tempo vs Danceability")
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_data, x='tempo', y='danceability', ax=ax)
ax.set_title(f"Tempo vs Danceability in {selected_subgenre}")
st.pyplot(fig)

# Visualization 2: Most Popular Artists/Albums
st.subheader("Most Popular Artists/Albums")
top_artists = filtered_data.groupby('track_artist')['track_popularity'].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
sns.barplot(x=top_artists.values, y=top_artists.index, ax=ax)
ax.set_title(f"Top 10 Artists in {selected_subgenre}")
st.pyplot(fig)

top_albums = filtered_data.groupby('track_album_name')['track_popularity'].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
sns.barplot(x=top_albums.values, y=top_albums.index, ax=ax)
ax.set_title(f"Top 10 Albums in {selected_subgenre}")
st.pyplot(fig)

# Visualization 3: Radar Chart
st.subheader("Radar Chart: Feature Averages for Selected Genre/Subgenre")
# Calculate average values for the radar chart
avg_features = filtered_data[['danceability', 'energy', 'loudness', 'speechiness', 
                              'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']].mean()

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
labels = avg_features.index
stats = avg_features.values

# Construct the radar chart
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
stats = np.concatenate((stats, [stats[0]]))
angles += angles[:1]

ax.fill(angles, stats, color='blue', alpha=0.25)
ax.plot(angles, stats, color='blue', linewidth=2)
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_title(f"Feature Averages for {selected_subgenre}")
st.pyplot(fig)

# Summary Section
st.subheader("Summary")
st.markdown(f"The visualizations above reflect the characteristics and popularity of songs within the '{selected_genre}' genre and the '{selected_subgenre}' subgenre. Use the dropdown menus to explore different genres and subgenres.")
