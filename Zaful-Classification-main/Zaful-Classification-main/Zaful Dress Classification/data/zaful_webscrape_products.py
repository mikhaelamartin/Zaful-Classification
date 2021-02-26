

# pip install bs4, requests NOT on Python

# gitbash: python shein_webscrape.py
from selenium import webdriver
import requests
from bs4 import BeautifulSoup
import re

baseurl = 'https://us.shein.com'
headers = {
	'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
}
items_page = 'https://us.shein.com/style/Floral-Dresses-FW-sc-00106552.html?icn=style&ici=us_tab01navbar05menu05dir11&srctype=category&userpath=category%3EWOMEN%3EDRESSES%3ESHOP%20BY%20TREND%3EFloral%20dresses&scici=navbar_WomenHomePage~~tab01navbar05menu05dir11~~5_5_11~~itemPicking_00106552~~SPcCccWomenCategory~~0~~50001'
productLinks = []

for productpage in range(1,2):

	if productpage != 1:
		addon = f'&page={productpage}'
		r = requests.get(items_page + addon)
	else:
		r = requests.get(items_page)

	# https://us.shein.com/style/Floral-Dresses-FW-sc-00106552.html?icn=style&ici=us_tab01navbar05menu05dir11&srctype=category&userpath=category%3EWOMEN%3EDRESSES%3ESHOP%20BY%20TREND%3EFloral%20dresses&scici=navbar_WomenHomePage~~tab01navbar05menu05dir11~~5_5_11~~itemPicking_00106552~~SPcCccWomenCategory~~0~~50001
	# https://us.shein.com/style/Floral-Dresses-FW-sc-00106552.html?icn=style&ici=us_tab01navbar05menu05dir11&srctype=category&userpath=category%3EWOMEN%3EDRESSES%3ESHOP%20BY%20TREND%3EFloral%20dresses&scici=navbar_WomenHomePage~~tab01navbar05menu05dir11~~5_5_11~~itemPicking_00106552~~SPcCccWomenCategory~~0~~50001&page=26
	productpagesoup = BeautifulSoup(r.content, 'html.parser')

	productList = productpagesoup.find_all('div', class_= re.compile(r'^c-goodsitem j-goodsli j-goodsli-.*col-xlg-20per col-lg-3 col-sm-4 col-xs-6 j-expose__content-goodsls$'))

	for product in range(1,2):
		# for link in product.find_all('a', href=True)
		productLink = baseurl + productList[0].find('a', class_='c-goodsitem__goods-name j-goodsitem__goods-name')['href']
		productLinks.append(baseurl + productLink) # link['href'])
		
		r = requests.get(productLink, headers=headers)

		productsoup = BeautifulSoup(r.content, 'lxml')

		name = productsoup.find('div', {'class' : 'product-intro__head-name'})
		print(name)

print(len(productLinks))
	# print(productLinks)