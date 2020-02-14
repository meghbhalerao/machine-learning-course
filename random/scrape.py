from urllib.request import urlopen
from bs4 import BeautifulSoup
html = urlopen("https://en.wikipedia.org/wiki/Hello")
soup = BeautifulSoup(html, 'html.parser')
print(soup.prettify())
