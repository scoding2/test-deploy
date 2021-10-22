import pandas as pd
import pickle
import requests
import json
import os

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('spotify_data.csv', index_col=0)

CLIENT_ID='d4036dcb0bfc4b18b322e5e332dc737c'
CLIENT_SECRET='58d1fdd2199e490cb03934748d25ab5a'


def spotconnect():
    """Returns headers for spotify api."""

    # get access token
    AUTH_URL = 'https://accounts.spotify.com/api/token'
    # POST
    auth_response = requests.post(AUTH_URL, {
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
    })

    # convert the response to JSON
    auth_response_data = auth_response.json()
    # save the access token
    access_token = auth_response_data['access_token']

    # GET song audio-features
    headers = {'Authorization': 'Bearer {token}'.format(token=access_token)}
    
    return headers


def get_nn_query(track_id):
    """Get spotify request for song audio-features, format it for query_nn()."""

    headers = spotconnect()
    
    r = requests.get('https://api.spotify.com/v1/audio-features/' + track_id, headers=headers)
    song_dict = r.json()
    
    feature_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'liveness', 'valence', 'tempo', 'duration_ms']
    
    # put audio attributes in same order as in the dataframe the estimator is fit to.
    query_nn = [song_dict[x] for x in feature_columns]

    return query_nn


def query_nn_pickles(song_features):
    """Load pickles, scale song_features, return 5 nearest neighbors."""
    # load pkls from current directory
    norm = pickle.load(open('norm.pkl', 'rb'))
    knn = pickle.load(open('nn.pkl', 'rb'))
    # scale features
    normed = norm.transform([song_features])
    # print(scaled)
    # get 5 nearest neighbors, returns a list of dataframe indices
    similar_five = knn.kneighbors(normed, 5, return_distance=False)

    return similar_five


def recomend(uri):
    """Take song_link, return 5 similar songs from dataframe."""
    # Slice uri out of spotify share link
    # uri = song_link[31:53]
    # Request song audio-features and format them for nearest-neighbors query
    features = get_nn_query(uri)
    # get nearest neighbors
    similar_songs = query_nn_pickles(features)

    # create links to spotify songs
    query_results = df.loc[similar_songs[0]]['url']
    art_tracks = df.loc[similar_songs[0]][['artist_name', 'track_name']].values
    links = query_results.tolist()
    
    # Wraps them together [artist, title, link]
    recommends = [[
        art_tracks[x][0], art_tracks[x][1],
        links[x]] for x in range(5)]
    return recommends, features


def search(query):
    """Searches for song using spotify search api."""

    headers = spotconnect()

    r = requests.get(
        'https://api.spotify.com/v1/search',
        headers=headers, params= {
            'q': query,
            'type': 'track',
            'limit': 1
        }
    )
    rs = r.json()

    importante = {
        'name': rs['tracks']['items'][0]['name'],
        'artist': rs['tracks']['items'][0]['album']['artists'][0]['name'],
        'album': rs['tracks']['items'][0]['album']['name'],
        'imageurl': rs['tracks']['items'][0]['album']['images'][2]['url'],
        'release': rs['tracks']['items'][0]['album']['release_date'],
        'url': rs['tracks']['items'][0]['external_urls']['spotify'],
        'id': rs['tracks']['items'][0]['id']
    }

    return importante