from urllib.request import urlopen as uRequest
from bs4 import BeautifulSoup as soup
import json
import re
import math

# name the output file to write to local disk
out_filename = "zaful_reviews.csv"
# header of csv file to be written
headers = "SKU,Individual Rating,Number of Pictures,Comment,Date Stamp,Color/Size,Overall Fit,Height,Waist,Hips,Bust\n"

# opens file, and writes headers
f = open(out_filename, "w", encoding="utf-8")
f.write(headers)


for page in range(1, 6): # loops over each page
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
	# floral_dresses = floral_dresses[8:11]
	
	for floral_dress in floral_dresses: # loops over items on page

		
		sku = str(floral_dress.find('strong', class_='my_shop_price')['data-sku'].strip())
		
		# # I N D I V I D U A L  D R E S S  P A G E 

		# grabs and parses through individual dress page
		indiv_floral_dress_page = floral_dress.find('p', class_='goods-title pr').a['href']
		print(indiv_floral_dress_page)
		second_uClient = uRequest(indiv_floral_dress_page)
		indiv_floral_dress_page_html = second_uClient.read()
		second_uClient.close()
		# html parser
		indiv_floral_dress_page_soup = soup(indiv_floral_dress_page_html, "html.parser")

		

		## L I S T  O F  R E V I E W S
		##

		does_review_page_exist = indiv_floral_dress_page_soup.find('footer', class_='footer pr tc clearfix')
		if (does_review_page_exist is not None):
			reviews_page = does_review_page_exist.find_all('a', href=True)
			if (reviews_page is not None):
				page_1 = reviews_page[0]['href']
				third_uClient = uRequest(page_1)
				reviews_page_html = third_uClient.read()
				third_uClient.close()
				# brings us to first review page
	
				# html parser
				reviews_page_soup = soup(reviews_page_html, "html.parser")
				total_reviews = reviews_page_soup.find('span', id='js-reviewTotal').text
				# pages remaining from first review page
				num_review_pages_left = math.ceil((int(total_reviews) - 10) / 10)

				# iterate over each review page
				for review_page in range(1, num_review_pages_left + 2):
					page_url = reviews_page[0]['href'][:-5] + f'-page-{review_page}' + '.html'
					fourth_uClient = uRequest(page_url)
					indiv_reviews_page_html = fourth_uClient.read()
					third_uClient.close()

					# html parser
					indiv_reviews_page_soup = soup(indiv_reviews_page_html, "html.parser")
					
					reviews = indiv_reviews_page_soup.findAll('div', class_='list-wrap')
					# iterate over each review in review page
					for indiv_review in reviews:
						stars = str(indiv_review.find('p', class_='star js-rate-star')['data-rate']) 
						if (indiv_review.find('div', class_='comment-pics') is not None):
							num_pics = str(len(indiv_review.find('div', class_='comment-pics').findAll('a'))) 
						else:
							num_pics = '0'
						if (indiv_review.find('p', class_='shopInfo f14 mt30') is not None):
							size = str(indiv_review.find('p', class_='shopInfo f14 mt30').findAll('span')).strip()
						else:
							size = 'N/A'
					
						comment = indiv_review.find('div', class_='comment-content fsb f16').text.strip().replace(",", "")
						date_stamp = indiv_review.find('p', class_='date').text.strip().replace(",", "")
						color_size = indiv_review.find('p', class_='attr').text.strip().replace(",", "")
						
						# if (indiv_review.find('p', class_='shopInfo f14 mt30').span is not None):
						#	overall_fit = indiv_review.find('p', class_='shopInfo f14 mt30').span.text.strip()
						# else:
						#	overall_fit = 'N/A'
						f.write(sku + "," + stars + ", " + num_pics + "," + comment + "," + date_stamp + "," + color_size + "," + size + "\n")

		elif (indiv_floral_dress_page_soup.find('ul', class_='list') is not None):
			only_one_review_page = indiv_floral_dress_page_soup.find('ul', class_='list')
			reviews = indiv_floral_dress_page_soup.findAll('li', class_='item js-reviewItem')
			for i in reviews:
				stars = str(only_one_review_page.find('dt', class_='star js-rate-star')['data-rate']) 			
				num_pics = str(len(only_one_review_page.findAll('dd', class_='photos clearfix logsss_event_ps')))
				if (only_one_review_page.find('p', class_='shopInfo f14 mt30') is not None):
					size = str(only_one_review_page.find('p', class_='shopInfo f14 mt30').findAll('span')).strip()
				else:
					size = 'N/A'
					
				comment = only_one_review_page.find('dd', class_='text').text.strip().replace(",", "")
				date_stamp = only_one_review_page.find('dd', class_='time').text.strip().replace(",", "")
				color_size = only_one_review_page.find('dd', class_='review-goods-item').text.strip().replace(",", "")
				f.write(sku + "," + stars + ", " + num_pics + "," + comment + "," + date_stamp + "," + color_size + "," + size + "\n")

f.close()
