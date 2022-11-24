import undetected_chromedriver as uc
from undetected_chromedriver.options import ChromeOptions
import re
import time

options = ChromeOptions()
options.add_argument('--disable-images')
options.add_argument('--disable-plugins')
options.add_argument('disable-translate')
driver = uc.Chrome(options=options)

url_list = ['https://bitinfocharts.com/comparison/bitcoin-price.html#alltime',
            'https://bitinfocharts.com/comparison/google_trends-btc.html#alltime',
            'https://bitinfocharts.com/comparison/activeaddresses-btc.html#alltime',
            'https://bitinfocharts.com/comparison/top100cap-btc.html#alltime']
for url in url_list:
    name = url.split('/')[-1].split('.')[0]

    # url = 'https://bitinfocharts.com/comparison/google_trends-btc.html#alltime'
    # url = 'https://bitinfocharts.com/comparison/transactions-btc.html'
    driver.get(url)
    time.sleep(2)

    obj = re.compile('\[new Date\(\"(?P<container>.*?)\]', re.S)
    results = obj.findall(driver.page_source)
    for result in results:
        s1, s2 = result.split('"),')
        with open(f'{name}.csv', mode='a', encoding=('utf-8-sig')) as f:
            f.write(','.join([s1, s2]))
            f.write('\n')
            f.flush()
            # print(s1, s2)
    print(f'{name}: done')

driver.close()
