"""
@author: sethp
@created: 09/10/2019
"""
from loadConfig import getToken

from sklearn import preprocessing, cluster, decomposition
import pandas as pd
import warnings
import spotipy
import time
import os

def getPlaylistTracks(spotifyConnection=getToken(), playlistName='Prabhat'):
    """
    :param spotifyConnection: an authenticated connection to spotify for a user
    :param playlistName: the name of the playlist
    :return: a pandas dataframe with information on all the tracks in the playlist
    """
    playlist = [i for i in spotifyConnection.user_playlists(spotifyConnection.me()['id'])['items'] if i['name'] == playlistName][0]
    tracks = spotifyConnection.user_playlist(spotifyConnection.me()['id'], playlist['id'])['tracks']
    items = tracks['items']
    while tracks['next']:
        tracks = spotifyConnection.next(tracks)
        items.extend(tracks['items'])
    return pd.DataFrame(items)

def dictCheck(col):
    """
    :param col: a column of a pandas dataframe
    :return: whether the column contains a dictionary or a list containing dictionaries
    """
    for row in col:
        if isinstance(row, dict):
            return 'dict'

        elif isinstance(row, list):
            try:
                if isinstance(row[0], dict):
                    return 'list'
            except:
                warnings.warn('empty list encountered :(')

        else:
            pass

def dfDictColumn(df, col, suffix):
    """
    :param df: a dataframe
    :param col: a column of the dataframe that just contains dictionaires (can be empty)
    :param suffix: suffix for the dictionary keys. generally use the column name
    :return: dataframe with dictionary items as columns. drops the original column
    """
    return df.join(pd.DataFrame(list(df[col])).add_suffix(suffix)).drop(col, axis=1)

def nestedDictToDf(df):
    """
    a function that loops over itself expanding all dictionaries found

    :requires: pd.DataFrame, pd.DataFrame.join(), pd.add_suffix(), pd.drop(), warnings.warn(), dfDictColumn()
    :param df: a dataframe with columns containing dictionaries or lists of dictionaries
    :return: a dataframe with all dictionary elements expanded out to columns
    """
    objectCols = df.select_dtypes(object).columns
    series = df[objectCols].apply(dictCheck, axis=0)
    dictCols = list(series.index[series == 'dict'])
    listCols = list(series.index[series == 'list'])

    if not dictCols and not listCols:
        return df

    for col in dictCols:
        suffix = '_' + col
        if df[col].isna().sum() == 0:
            df = dfDictColumn(df, col=col, suffix=suffix)
        else:
            warnings.warn('Rows in the dataframe have been replaced with an empty dictionary')
            df[col] = df[[col]].applymap(lambda x: {} if pd.isnull(x) else x)
            df = dfDictColumn(df, col=col, suffix=suffix)

    for col in listCols:
        subset = df[col]
        lsDict = [{'{0}_{1}'.format(idx, col) : v for idx, v in enumerate(row)} for row in subset]
        lsDf = pd.DataFrame(lsDict).applymap(lambda x: {} if pd.isnull(x) else x)
        df = df.join(lsDf).drop(col, axis=1)

    return nestedDictToDf(df)

def audioFeatures(df, spotifyConnection=getToken(), suffix='_audiofeat'):
    """
    :param df: a base dataframe with track information
    :param spotifyConnection: a connection to spotify for a particular user
    :return: the df with audio features for each track joined to it
    """
    audioFeatures = [spotifyConnection.audio_features(df['uri_track'][i:i+50]) for i in range(0, df.shape[0], 50)]
    audioFeatures = pd.DataFrame([item for sublist in audioFeatures for item in sublist])
    audioColumns = audioFeatures.select_dtypes(exclude=object).columns
    return df.join(audioFeatures, rsuffix=suffix), audioColumns

def timeSinceAdded(df):
    df['added_at'] = pd.DatetimeIndex(df['added_at']).tz_localize(None)
    df['timeSinceAdded'] = (pd.to_datetime('today') - df['added_at']).dt.days
    return df

def getInputData():
    tracksDf = getPlaylistTracks()
    tracksDf = nestedDictToDf(tracksDf)
    tracksDf, audioColumns = audioFeatures(df=tracksDf)
    tracksDf = timeSinceAdded(tracksDf)
    return tracksDf, audioColumns

def loopOverTracksForLabelling(dfName):
    tracksDf = pd.read_csv(os.getcwd() + f'\\{dfName}.csv')
    spotifyConnection = getToken()

    for idx, row in tracksDf.iterrows():
        trackUri = row['uri_track']
        spotifyConnection.start_playback(uris=[trackUri], position_ms=30000)
        time.sleep()


## clustering ----------------------------------------------------------------------------------------------------------
#cap and floor your data to remove outliers (could also just remove outliers)
#cluster size should be between 5-30% of population

def PCA(x):
    pca = decomposition.PCA(n_components=1, random_state=13)
    x = pca.fit_transform(x)
    return x


def clustering(x, cols, scaler, method='kmeans'):
    methods = {'kmeans': cluster.k_means}
    df = x.copy()
    df = pd.DataFrame(scaler.fit_transform(df[cols]), columns=cols)

    method = methods[method]
    kMeans = method(df, n_clusters=5, init='k-means++', precompute_distances=True, random_state=13,
                             copy_x=True, algorithm='full')
    return kMeans[1]

def createPlaylists(df, audioColumns, scaler, method='kmeans', spotifyConnection=getToken(), stepSize=100):
    results = clustering(df, audioColumns, scaler)
    df['label'] = results

    labels = df['label'].unique()
    today = pd.to_datetime('today')

    for label in labels:
        name = '{label}_{today}'.format(label=label, today=today)
        spotifyConnection.user_playlist_create(user= spotifyConnection.me()['id'], name=name, public=False)
        playlistId = [playlist for playlist in spotifyConnection.user_playlists(spotifyConnection.me()['id'])['items']
                      if playlist['name'] == name][0]['id']

        tracks = df['uri'][df['label'] == label].to_list()
        for i in range(0, len(tracks), stepSize):
            spotifyConnection.user_playlist_add_tracks(user=spotifyConnection.me()['id'], playlist_id=playlistId,
                                                       tracks=tracks[i:i+stepSize])

    print('This is a playlist created using python clustering. \n' \
          'Scaler used: {scaler} \n' \
          'Method used: {method}'.format(scaler=str(type(scaler)), method=method))

## ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # scaler1 = preprocessing.MinMaxScaler()
    # scaler2 = preprocessing.StandardScaler()
    # scaler3 = preprocessing.PowerTransformer(method='yeo-johnson')
    # scaler4 = preprocessing.QuantileTransformer()
    #
    # tracksDf, audioColumns = getInputData()
    # createPlaylists(tracksDf, audioColumns, scaler4)
    loopOverTracksForLabelling(dfName='cluster1')