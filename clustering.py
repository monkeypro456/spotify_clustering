import spotipy
import spotipy.util as util
import pandas as pd
import warnings
from sklearn import preprocessing, cluster, decomposition

path = 'C:/Users/sethp/Documents/Documents/spotifyresults.csv'
pd.set_option('display.max_columns',500)

def getToken(scope='playlist-read-private user-read-recently-played user-top-read app-remote-control streaming',
             username='monkeypro456'):
    '''
    :param scope: the scopes that we are intested in
    :param username: the user that we want data for
    :return: an authenticated token for connection to a spotify account
    '''
    token = util.prompt_for_user_token(
                username=username,
                scope=scope,
                client_id='64e451339e1049acabac90122ddf39ec',
                client_secret='3ae748405530465eae9acebeaab798ae',
                redirect_uri='http://localhost/')
    return spotipy.client.Spotify(auth=token)

def getPlaylistTracks(spotifyConnection=getToken(), playlistName='Prabhat'):
    '''
    :param spotifyConnection: an authenticated connection to spotify for a user
    :param playlistName: the name of the playlist
    :return: a pandas dataframe with information on all the tracks in the playlist
    '''
    playlist = [i for i in spotifyConnection.user_playlists('monkeypro456')['items'] if i['name'] == playlistName][0]
    tracks = spotifyConnection.user_playlist(spotifyConnection.me()['id'], playlist['id'])['tracks']
    items = tracks['items']
    while tracks['next']:
        tracks = spotifyConnection.next(tracks)
        items.extend(tracks['items'])
    return pd.DataFrame(items)

def dictCheck(col):
    '''
    :param col: a column of a pandas dataframe
    :return: whether the column contains a dictionary or a list containing dictionaries
    '''
    #TODO: tried to check if the column is an object but was not doing it correctly :(
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
    '''
    :param df: a dataframe
    :param col: a column of the dataframe that just contains dictionaires (can be empty)
    :param suffix: suffix for the dictionary keys. generally use the column name
    :return: dataframe with dictionary items as columns. drops the original column
    '''
    return df.join(pd.DataFrame(list(df[col])).add_suffix(suffix)).drop(col, axis=1)

def nestedDictToDf(df):
    '''
    a function that loops over itself expanding all dictionaries founc
    :requires: pd.DataFrame, pd.DataFrame.join(), pd.add_suffix(), pd.drop(), warnings.warn(), dfDictColumn()
    :param df: a dataframe with columns containing dictionaries or lists of dictionaries
    :return: a dataframe with all dictionary elements expanded out to columns
    '''
    '''
    dictCols, listCols = [], []
    objectCols = df.dtypes[df.dtypes == object].index
    for col in objectCols: # loop over all columns checking for dictionaries and lists containing dictionaires
        for row in df[col]:

            if isinstance(row, dict):
                dictCols.extend([col])
                break # if it is a dicitionary break

            if isinstance(row, list):
                if row:
                    if isinstance(row[0], dict):
                        listCols.extend([col])
                        break
    '''
    objectCols = df.select_dtypes(object).columns
    series = df[objectCols].apply(dictCheck)
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
    '''
    :param df: a base dataframe with track information
    :param spotifyConnection: a connection to spotify for a particular user
    :return: the df with audio features for each track joined to it
    '''
    audioFeatures = [spotifyConnection.audio_features(df['uri_track'][i:i+50]) for i in range(0, df.shape[0], 50)]
    audioFeatures = pd.DataFrame([item for sublist in audioFeatures for item in sublist])
    audioColumns = audioFeatures.select_dtypes(exclude=object).columns
    return df.join(audioFeatures, rsuffix=suffix), audioColumns

def timeSinceAdded(df):
    df['added_at'] = pd.DatetimeIndex(tracksDf['added_at']).tz_localize(None)
    df['timeSinceAdded'] = (pd.to_datetime('today') - tracksDf['added_at']).dt.days
    return df

def testing(col):
    return col.dtype ### WHY DOES THIS NOT WORK. TRY FOR ONE COLUMN AND THEN THE WHOLE DATEFRAME`

tracksDf = getPlaylistTracks()
tracksDf = nestedDictToDf(tracksDf)
tracksDf, audioColumns = audioFeatures(df=tracksDf)
tracksDf = timeSinceAdded(tracksDf)

## clustering ----------------------------------------------------------------------------------------------------------
#cap and floor your data to remove outliers (could also just remove outliers)
#cluster size should be between 5-30% of population

def PCA(x):
    pca = decomposition.PCA(n_components=1, random_state=13)
    x = pca.fit_transform(x)
    return x

def clustering(x, cols, scaler):
    df = x.copy()
    df = pd.DataFrame(scaler.fit_transform(df[cols]), columns=cols)

    kMeans = cluster.k_means(df, n_clusters=5, init='k-means++', precompute_distances=True, random_state=13,
                             copy_x=True, algorithm='full')
    return kMeans[1]

#minMaxScaler = preprocessing.MinMaxScaler()
#standardiserScaler = preprocessing.StandardScaler()
#powerScaler = preprocessing.PowerTransformer(method='yeo-johnson')
quantileScaler = preprocessing.QuantileTransformer()
results = clustering(tracksDf, audioColumns, quantileScaler)
tracksDf['label'] = results
