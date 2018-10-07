import urllib.request
import ssl


def fetch(url):
    try:
        request = urllib.request.Request(url)
        response = urllib.request.urlopen(request, context=ssl._create_unverified_context(), timeout=10)
        return response
    #  ignore errors
    except urllib.error.URLError:
        pass

    except ValueError:
        pass

    except Exception:
        pass


def fetch_url(url):
    return fetch(url).geturl()
