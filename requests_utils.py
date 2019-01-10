import requests
from bs4 import BeautifulSoup

def fetch(url, **kwargs):
    try:
        response = requests.get(url, kwargs)
        return response
    #  ignore errors
    # except Error:
    #     pass

    except ValueError:
        pass

    except Exception:
        pass


def fetch_url(url, **kwargs):
    response = fetch(url, **kwargs)

    if response:
        if response.status_code == 403:
            return response.history[1].url

        elif response.status_code == 200:
            url = response.url if response.url else url
            return url
    else:
        pass
