# import urllib.request
import requests
# import ssl


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


def fetch_url(url):
    return fetch(url).geturl()
