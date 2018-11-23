import requests


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
    response = fetch(url)

    if response:
        url = response.history[0].url if response.history else response.url
        return url
    else:
        pass
