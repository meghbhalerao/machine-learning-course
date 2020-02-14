from googlesearch import search
for url in search('machine learning', tld='com.pk', lang='es', stop=5):
        print(url)
