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


def fetch_url(url, **kwargs):
    response = fetch(url, **kwargs)

    if response:
        url = response.url if response.url else url
        return url

    else:
        pass
