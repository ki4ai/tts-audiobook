import requests
import itertools
import time

s = requests.session()

ip_address = 'http://0.0.0.0:8081'

aa = time.time()
dataset = 'M1.4/TTS_INPUT'
r = s.post(ip_address, data={'dataset': dataset})
print('It takes {}s'.format(time.time() - aa))
