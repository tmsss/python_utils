from botometer import Botometer
import apis
import file_utils as fx
from tweepy import TweepError

twitter_api = apis.twitter_api()
botometer_key = apis.botometer_api()

twitter_app_auth = {
    'consumer_key': twitter_api['consumer_key'],
    'consumer_secret': twitter_api['consumer_secret'],
    'access_token': twitter_api['access_token'],
    'access_token_secret': twitter_api['access_secret'],
}

'''
    Botometer implementation from https://github.com/IUNetSci/botometer-python

    Details at https://rapidapi.com/OSoMe/api/Botometer%20Pro/details
'''

bom = Botometer(wait_on_ratelimit=True, rapidapi_key=botometer_key, **twitter_app_auth)


# Check a single account by screen name or id
def get_user_account(user, id=True):

    if id:
        user = int(user)

    return bom.check_account(user)

def get_accounts(users, folder, id=True):

    collected = fx.get_fnames(folder)

    for user in users:
        if user not in collected:
            try:
                result = (user, get_user_account(user))
                fx.save_pickle('{}/{}'.format(folder, user), result)
            
            except TweepError as e:
                print('Could not retrieve info for user: {}'.format(user))
                print(e)
                # print('exception raised, waiting 15 minutes')
                # time.sleep(15*60)
                pass

