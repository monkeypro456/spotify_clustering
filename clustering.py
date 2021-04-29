"""
@author: sethp
@created: 09/10/2019
"""
from loadConfig import getToken
from sklearn import preprocessing, cluster, decomposition
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import numpy as np
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
        time.sleep(7)

## graphical analysis --------------------------------------------------------------------------------------------------
def plotYearlyDifferences(audioColumns):
    try:
        tracksDf = pd.read_csv(os.getcwd() + '\\tracksDf.csv')
        tracksDf['added_at'] = pd.to_datetime(tracksDf['added_at'])
        tracksDf = tracksDf[audioColumns + ['added_at','uri','name_track']]
    except FileNotFoundError:
        tracksDf, audioColumns = getInputData()

    tracksDf['added_year'] = tracksDf['added_at'].dt.year
    for year in tracksDf['added_year'].unique():
        yearDf = tracksDf[tracksDf['added_year'] == year]

        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(nrows=1, ncols=9)
        fig.suptitle(f'Violin plots for audio features {year}')
        for idx, col in enumerate(audioColumns):
            string = f'ax{idx + 1}.violinplot(yearDf["{col}"], showmedians=True)'
            string2 = f'ax{idx + 1}.set_title("{col}")'
            string3 = f'ax{idx + 1}.set_xticks([1])'
            string4 = f'ax{idx + 1}.set_xticklabels(["{col}"])'
            exec(string), exec(string2), exec(string3), exec(string4)
    plt.show()

def plotLabelCounts(idDict):
    clusterDf = pd.read_csv(os.getcwd() + '\\allClusters.csv')
    clusterCounts = clusterDf['cluster_id'].value_counts()
    clusterCounts = clusterCounts.rename(index=idDict)

    fig, ax = plt.subplots()
    ax.set_xlabel('Cluster names')
    ax.set_ylabel('Cluster counts')
    fig.suptitle('Bar chart showing counts for labelled clusters')
    pps = ax.bar(clusterCounts.index, clusterCounts)
    for p in pps:
        height = p.get_height()
        perc = height / sum(clusterCounts)
        ax.annotate(f'({height}, {perc:.2f})', xy=(p.get_x() + p.get_width() / 2, height), xytext=(0,3),
                    textcoords='offset points', ha='center', va='bottom')
    plt.show()

def createPCAanalysis(audioColumns):
    tracksDf = pd.read_csv(os.getcwd() + '\\tracksDf.csv')
    _, pca, _ = applyPCA(tracksDf, audioColumns)

    fig, ax = plt.subplots()
    ax.plot(range(1, len(audioColumns) + 1), pca.explained_variance_ratio_, 'ro-', linewidth=2)
    fig.suptitle('Scree plot')
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Proportion of variance explained')
    for idx, cumVal in enumerate(np.cumsum(pca.explained_variance_ratio_)):
        ax.annotate(f'{cumVal:.2f}', xy=(idx + 1, pca.explained_variance_ratio_[idx]), xytext=(0,3),
                    textcoords='offset points', ha='center', va='bottom')
    plt.show()

## clustering ----------------------------------------------------------------------------------------------------------
#cap and floor your data to remove outliers (could also just remove outliers)
#cluster size should be between 5-30% of population

def applyPCA(tracksDf, audioColumns, numComponents=None):
    scaler = preprocessing.StandardScaler()
    scaledDf = pd.DataFrame(scaler.fit_transform(tracksDf[audioColumns]), columns=audioColumns)
    pca = decomposition.PCA(n_components=len(audioColumns))
    principalComponents = pd.DataFrame(pca.fit_transform(scaledDf),
                                       columns=[f'Principal component {i + 1}' for i in range(len(audioColumns))])
    if numComponents is not None:
        principalComponents = principalComponents[[f'Principal component {i + 1}' for i in range(numComponents)]]
    return principalComponents, pca, scaler

def chooseNumClustersAnalysis(audioColumns, numComponents, numTestClusters):
    tracksDf = pd.read_csv(os.getcwd() + '\\tracksDf.csv')
    principalComponents, pca, scaler = applyPCA(tracksDf, audioColumns, numComponents)

    sumSquDiff = []
    for k in range(1, numTestClusters + 1):
        km = cluster.KMeans(n_clusters=k, init='k-means++', random_state=13)
        km = km.fit(principalComponents)
        sumSquDiff.append(km.inertia_)

    plt.plot(range(1, numTestClusters + 1), sumSquDiff, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of squared distances')
    plt.title('Elblow method for finding optimal k')
    plt.show()

def make_spider(row, title, color, df):
    # number of variable
    categories = df.columns
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(2, 3, row + 1, polar=True, )

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([-2, -1, 0, 1, 2], ["-2", "-1", "0", "1", "2"], color="grey", size=7)
    plt.ylim(-3, 3)

    # Ind1
    values = df.iloc[row].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    plt.title(title, size=11, color=color, y=1.1)

def createSpiderPlot(df, groupbyColumn, columnNames, myDpi=200, scaler=None):
    grouped = df.groupby(by=[groupbyColumn]).mean()
    if scaler is None:
        scaler = preprocessing.StandardScaler()
        grouped = pd.DataFrame(scaler.fit_transform(grouped), columns=columnNames)
    else:
        grouped = pd.DataFrame(scaler.transform(grouped), columns=columnNames)
    plt.figure(figsize=(1000/myDpi, 1000/myDpi), dpi=myDpi)
    myPalette = plt.cm.get_cmap("Set2", len(grouped.index))
    for row in range(0, len(grouped.index)):
        make_spider(row, title=f'Cluster {row}', color=myPalette(row), df=grouped)
    plt.show()
    return scaler

def analyseClusters(audioColumns, numComponents, method, k=None):
    trainPrediction, testPrediction = applyClustering(audioColumns, numComponents, method, k=k, testing=True)

    columnNames = [col[:3] for col in audioColumns]
    _ = createSpiderPlot(trainPrediction, 'cluster_id', columnNames)
    scaler = createSpiderPlot(testPrediction.drop('cluster_id', axis=1), 'prediction_cluster_id', columnNames)
    _ = createSpiderPlot(testPrediction.drop('prediction_cluster_id', axis=1), 'cluster_id', columnNames, scaler=scaler)

def applyClustering(audioColumns, numComponents, method='kmeans', k=None, testing=False):
    tracksDf = pd.read_csv(os.getcwd() + '\\tracksDf.csv')
    if testing:
        clusterIDDf = pd.read_csv(os.getcwd() + '\\allClusters.csv')
        clusterIDDf['cluster_id'] = clusterIDDf['cluster_id'].astype(int)
        tracksDf = tracksDf.merge(clusterIDDf[['cluster_id', 'id_track']], how='left', on='id_track')
        trainDf = tracksDf[tracksDf['cluster_id'].isna()]
    else:
        trainDf = tracksDf

    principalComponents, pca, scaler = applyPCA(trainDf, audioColumns, numComponents)
    methods = {'kmeans': cluster.KMeans(n_clusters=k, init='k-means++', random_state=13)}
    method = methods[method]
    method = method.fit(principalComponents)
    trainPrediction = method.predict(principalComponents)
    trainPrediction = trainDf[audioColumns].assign(cluster_id=trainPrediction)

    if testing:
        testDf = scaler.transform(clusterIDDf[audioColumns])
        testDf = pd.DataFrame(pca.transform(testDf),
                              columns=[f'Principal component {i + 1}' for i in range(len(audioColumns))])
        testDf = testDf[[f'Principal component {i + 1}' for i in range(numComponents)]]
        testPrediction = method.predict(testDf)
        testPrediction = clusterIDDf[audioColumns + ['cluster_id']].assign(prediction_cluster_id=testPrediction)
    else:
        testPrediction = None
        trainPrediction = trainPrediction.assign(uri=tracksDf['uri'])
    return trainPrediction, testPrediction

def createPlaylists(audioColumns, numComponents, method='kmeans', k=6, spotifyConnection=getToken(), stepSize=100):
    clusters, _ = applyClustering(audioColumns, numComponents, method, k, testing=False)

    labels = clusters['cluster_id'].unique()
    today = str(pd.to_datetime('today').date())

    for label in labels:
        name = f"Playlist from python clustering - cluster {label} on {today}"
        spotifyConnection.user_playlist_create(user= spotifyConnection.me()['id'], name=name, public=False)
        playlistId = [playlist for playlist in spotifyConnection.user_playlists(spotifyConnection.me()['id'])['items']
                      if playlist['name'] == name][0]['id']

        tracks = clusters['uri'][clusters['cluster_id'] == label].to_list()
        for i in range(0, len(tracks), stepSize):
            spotifyConnection.user_playlist_add_tracks(user=spotifyConnection.me()['id'], playlist_id=playlistId,
                                                       tracks=tracks[i:i+stepSize])

## ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    cols = ['acousticness', 'danceability', 'energy',
       'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
    dic = {1.0: 'Pop music upbeat', 2.0: 'Pop music alternative', 3.0: 'Pop music low-key/sad', 4.0: 'Electronic/dance',
           5.0: 'Rock', 6.0: 'Metal'}
    createPlaylists(cols, 5)
    #analyseClusters(cols, 5, 'kmeans', 6)
    #applyClustering(cols, 5, k=6)
    #chooseNumClustersAnalysis(cols, 5, 10)
    #plotLabelCounts(dic)
    #tracksDf, audioColumns = getInputData()
    #plotYearlyDifferences(cols)
    # createPlaylists(tracksDf, audioColumns, scaler)
    #loopOverTracksForLabelling(dfName='cluster1')