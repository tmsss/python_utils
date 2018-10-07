from apiclient.discovery import build
# from apiclient.errors import HttpError
import requests
from bs4 import BeautifulSoup
import urllib.request
import json
import os
import re
from utils import file_utils as fx


headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:49.0) Gecko/20100101 Firefox/49.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }


# youtube
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Google Custom Search from https://cse.google.com/
CSE_API_SERVICE_NAME = "customsearch"
CSE = '000060482974546939694:tkonyg_kluk'
CSE_API_VERSION = "v1"


def youtube_service():
    return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)


def cse_service():
    return build(CSE_API_SERVICE_NAME, CSE_API_VERSION, developerKey=DEVELOPER_KEY)


# parameters at https://developers.google.com/apis-explorer/#p/customsearch/v1/search.cse.list
def get_search(search_term, **kwargs):
    try:
        res = cse_service().cse().list(q=search_term, cx=CSE, **kwargs).execute()
        return res['items']
    except Exception:
        pass


def get_search_scraper(q):
    s = requests.Session()
    q = '+'.join(q.split())
    url = 'https://www.google.com/search?q=' + q + '&ie=utf-8&oe=utf-8'
    r = s.get(url, headers=headers)

    soup = BeautifulSoup(r.text, "html.parser")
    output = []
    for searchWrapper in soup.find_all('h3', {'class':'r'}): #this line may change in future based on google's web page structure
        url = searchWrapper.find('a')["href"]
        text = searchWrapper.find('a').text.strip()
        result = {'text': text, 'url': url}
        output.append(result)

    return output


def get_broken_link(url, domain):
    try:
        # search for url in google for specified url domain
        response = get_search(url, siteSearch=domain)

        # return the complete link if found and the boolean for offline
        for result in response:
            if result['link'].find(url) > -1:
                return (result['link'], False)
            else:
                return (url, True)
    except Exception:
        return (url, True)


def get_video_id(url):

    # parser para linkis
    if re.search('linkis.com/www.youtube.com', url):
        url = parser_linkis(url)

    url = re.sub(r"http.?://", '', url)

    print('Formating url: ' + url)

    if re.search('youtu.be', url):
        return url.split('/')[1]

    elif re.search('=', url):
        return url.split('=')[1]


def parser_linkis(url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',
        }

        try:
            request = urllib.request.Request(url, headers=headers)
            response = urllib.request.urlopen(request)
            data = response.read()

            bs = BeautifulSoup(data,"html.parser")

            return bs.find("meta", {"name":"twitter:app:url:googleplay"})['content']

        except urllib.error.URLError as e:
            print('Error: ' + str(e.reason))

        except ValueError:
            pass

        except Exception:
            pass


def get_video_info(video_id, parser=True):
    if parser:
        video_id = get_video_id(video_id)

    video_response = youtube_service().videos().list(
        id=video_id,
        part='snippet, recordingDetails, statistics'
    ).execute()

    return video_response


def save_video_info(video_id, parser, folder, fname):
    try:
        info = get_video_info(video_id, parser=parser)

        fname = os.getcwd() + folder + '/' + fname
        dirname = os.path.dirname(fname)

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if len(info['items']) > 0:
            with open(fname + '.json', 'w') as f:
                json.dump(info['items'], f)
        else:
            fx.pickle_list(fname, 'videos', 'offline_videos.pkl')

    except Exception as e:
        if video_id is not None:
            print('Error in: ' + video_id)
            fx.pickle_list(fname, 'videos', 'offline_videos.pkl')
        pass


def get_video_ids_folder(folder):
    files = fx.get_filepaths(folder)
    files = [f.strip(folder + '\\').strip('.json') for f in files]
    return files
