## Credits to
## 
## Arham Akheel authored 2 years ago

# Code inspired by and taken from: 

# https://www.youtube.com/watch?v=XQgXKtPSzUI&list=PL8eNk_zTBST-SaABhXwBFbKvvA0tlRSRV
# https://www.youtube.com/watch?v=nCuPv3tf2Hg

## 
from urllib.request import urlopen as uRequest
from bs4 import BeautifulSoup as soup
import json
import re


# name the output file to write to local disk
out_filename = "zaful_floral_dresses.csv"
# header of csv file to be written
headers = "Rank,SKU,Item Name,Shop Price,Recommended Retail Price,Deals,Available Colors,Overall Rating,Number of Reviews,True Fit Percentage,Too Small Percentage,Too Large Percentage\n"

# opens file, and writes headers
f = open(out_filename, "w", encoding="utf-8")
f.write(headers)


for page in range(1, 6): # loops over each page
	num_products = 0
	sku = ''
	rank = ''
	name = ''
	shop_price = ''
	rrp = ''
	deal_str = ''
	sale_str = ''
	rating = ''
	review_total = ''
	true_fit_percentage = ''
	too_small_percentage = ''
	too_large_percentage = ''
	review_info_str = ''
	color_str = ''
	if page == 1: # page 1 has different url format
		my_url = 'https://www.zaful.com/w/floral-dresses/e_5/'
	else:
		my_url = f'https://www.zaful.com/w/floral-dresses/e_5/g_{page}.html'

	# opening up connection, grabbing page, then close
	first_uClient = uRequest(my_url)
	page_html = first_uClient.read()
	first_uClient.close()
	# html parser
	floral_dresses_soup = soup(page_html, "html.parser")
	# grabs each product
	floral_dresses = floral_dresses_soup.findAll("li", {"class":"js_proList_item logsss_event_ps"})

	# floral_dresses = floral_dresses[0:15]
	for floral_dress in floral_dresses: # loops over items on page

		js_data = json.loads(floral_dress.div.a.img['data-logsss-browser-value'].replace("\'", "\""))
		rank = str(js_data['bv']['rank'])
		sku = str(floral_dress.find('strong', class_='my_shop_price')['data-sku'].strip())
		name = str(floral_dress.div.a['title'].strip())
		shop_price = str(floral_dress.find('strong', class_='my_shop_price')['data-orgp'].strip())
		rrp = str(floral_dress.find('strong', class_='my_shop_price js_market_wrap')['data-orgp'].strip())
		# rrp_str_present = floral_dress.find('strong', 'data-orgp'= f'{rrp}').string

		# mysale = floral_dress.find('span', class_=re.compile(r"^sale-flag js-price"))
		# mysale = floral_dress.find('span', class_='^sale-flag js-price-')
		# sale_str = str(mysale) # 'Viewable' if mysale is not  else "No viewable sale"


		mydeal = floral_dress.find('div', class_='tip-one')
		if mydeal is not None:
			deal_str = mydeal.text.strip()
		else:
			deal_str = "No viewable deal"

		# colors
		colors_list = []
		# self_color_if = floral_dress.find('div', class_='block js_list_color_switch cur logsss_event_hover_cl')['data-color']
		self_color_list = floral_dress.div.a['title'].split('- ')[1:]
		joint = ' '
		self_color = joint.join(self_color_list).split(" ")[0:-1][0]
		colors_list.append(self_color)
		color_occurences = floral_dress.findAll('div', class_='block js_list_color_switch logsss_event_hover_cl')
		num_colors = len(color_occurences)

		for i in range(0, num_colors):
			colors_list.append(color_occurences[i]['data-color'])

		joint = ', '
		colors_str = str(joint.join(colors_list))


		# # I N D I V I D U A L  D R E S S  P A G E 

		# grabs and parses through individual dress page
		indiv_floral_dress_page = floral_dress.find('p', class_='goods-title pr').a['href']
		print(indiv_floral_dress_page)
		second_uClient = uRequest(indiv_floral_dress_page)
		indiv_floral_dress_page_html = second_uClient.read()
		second_uClient.close()
		# html parser
		indiv_floral_dress_page_soup = soup(indiv_floral_dress_page_html, "html.parser")

		rating = indiv_floral_dress_page_soup.find('p', class_='js-rate-all ml10')
		rating = 'N/A' if rating is None else str(rating.text)

		review_total = indiv_floral_dress_page_soup.find('span', class_='js-review-count')
		review_total = 'N/A' if review_total is None else str(review_total.text)

		# review_total = str(indiv_floral_dress_page_soup.findAll('span', class_='js-review-count')[0].text)
		too_small_percentage = indiv_floral_dress_page_soup.find('span', class_='js-overall-small fr')
		too_small_percentage = 'N/A' if too_small_percentage is None else str(too_small_percentage.text)

		true_fit_percentage = indiv_floral_dress_page_soup.find('span', class_='js-overall-fit fr')
		true_fit_percentage = 'N/A' if true_fit_percentage is None else str(true_fit_percentage.text)

		too_large_percentage = indiv_floral_dress_page_soup.find('span', class_='js-overall-large fr')
		too_large_percentage = 'N/A' if too_large_percentage is None else str(too_large_percentage.text)

		f.write(rank + "," + sku + "," + name.replace(",", " ") + "," + shop_price + "," + 
						rrp + "," + deal_str.replace(",", " ") + "," + colors_str.replace(",", ";") + "," +
						rating + "," + review_total + "," + true_fit_percentage + "," + too_small_percentage + "," + 
						too_large_percentage + "\n") 

		num_products += 1				
f.close()
print(num_products)