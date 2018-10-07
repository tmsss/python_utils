import re


def get_url(url, **kwargs):
    # sep = '\\'  # remove everything after separator
    # url = str(url.encode('utf-8')).split(sep, 1)[0]
    try:
        url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', url)
        if 'strip' in kwargs:
            return str(url).strip(r'^\[".|(\\\\xa0)|(\\\\xc)|(\\\\xe2)|(\\\\u206)|()|\'\]\"\]$')
        else:
            return url[0]
    except Exception as e:
        print('Error in:' + str(url))
        print(e)
        pass


def find_domain(url):
    try:
        return str(url).split("//")[-1].split("/")[0].split('?')[0]

    except Exception as e:
        print('Error in:' + str(url))
        print(e)
        pass


def find_subdomain(url):
    # remove http:// and https://
    url = re.sub(r"http.?://", '', url)

    # remove empy spaces
    url = re.sub(r" ", '', url)

    url = url.split('?')[0]

    url = re.sub(r".php", '', url)

    folder = (url.count('/'))

    if folder == 1:
        return url
    else:
        return "/".join(url.split('/', 2)[:2])


def check_domain(url, domain):
    if domain in find_domain(url).split('.'):
        return True
    else:
        return False


def set_query_from_url(url):
    url = re.sub(r"http.?://", '', url)

    query = ' '.join(re.split('/|-|\.', url))

    return query


def strip_urls(text):
    try:
        if type(text) == str:
            # remove urls with and without http(s) or www
            text = re.sub(r'[^ ]+\.[^ ]+', '', text, flags=re.MULTILINE)
            return text
        else:
            pass
    except ValueError:
        pass
