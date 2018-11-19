import urllib2, requests
# from urllib import urlencode
from bs4 import BeautifulSoup
# from urlparse import urlunparse
import random
import cookielib



class Fetcher(object):

    def __init__(self, url, query):
        self.url = url
        self.query = query

    def LoadUserAgents(self):
        uafile = "ua.txt"
        uas = []

        with open(uafile, 'rb') as uaf:
            for ua in uaf:
                if ua:
                    uas.append(ua.strip())
        random.shuffle(uas)

        return uas

    def get_response(self, url):
        cookieJar = cookielib.CookieJar()
        uas = self.LoadUserAgents()
        ua = random.choice(uas)

        headers = [
            # ('Host', "twitter.com"),
            ('User-Agent', str(ua)),
            ('Accept', "application/json, text/javascript, */*; q=0.01"),
            ('Accept-Language', "en-US;q=0.7,en;q=0.3"),
            ('X-Requested-With', "XMLHttpRequest"),
            ('Referer', url),
            ('Connection', "keep-alive")
        ]

        try:
            opener = urllib2.build_opener(
                urllib2.HTTPCookieProcessor(cookieJar))
            opener.addheaders = headers
            response = opener.open(url)
            return response.read()

        except Exception as e:
            print("Error response: %s" % e)
            # verify rejected user agents
            print("ua rejected: %s" % ua)
