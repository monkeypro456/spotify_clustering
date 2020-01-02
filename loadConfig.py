import spotipy.util as util
import spotipy
import yaml
import os

ENVIRONMENTSETTINGSFILEPATH = os.path.join(os.getcwd(), 'EnvironmentSettingsSpotify.yaml')

def getToken():
    '''
    :return: an authenticated token for connection to a spotify account
    '''
    with open(ENVIRONMENTSETTINGSFILEPATH, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    token = util.prompt_for_user_token(**data)
    return spotipy.client.Spotify(auth=token)