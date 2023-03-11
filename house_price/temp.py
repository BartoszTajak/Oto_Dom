from bs4 import BeautifulSoup as bs
import requests
link = r'https://www.otodom.pl/pl/oferta/mieszkanie-2-pokojowe-taras-ogrodek-garaz-ID4kBBS'
r = requests.get(link)
soup = bs(r.content, 'html.parser')


adds_no = soup.find_all('div', {'class': 'css-1h52dri estckra7'})


print(adds_no)







