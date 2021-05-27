
from flask import render_template, url_for, flash, redirect
import app
from forms import SearchForm
from models import Book
import pandas as pd
import numpy as np
import pickle


book_tags_cleaned = pd.read_csv(r'book_tags_cleaned.csv')
popular_list = pd.read_csv(r'popular_list.csv')

# indexes of popular books
recommended_books = np.arange(0,12)
popular_books = [{'title': popular_list.iloc[i, 1],
		'isbn': popular_list.iloc[i, 2],
		'author': popular_list.iloc[i, 3],
		'img': popular_list.iloc[i, -2],
		'date_published':popular_list.iloc[i, 4]} for i in recommended_books]
books_stats2 = [{'title': popular_list.iloc[i, 1],
		'isbn': popular_list.iloc[i, 2],
		'author': popular_list.iloc[i, 3],
		'img': popular_list.iloc[i, -2],
		'date_published':popular_list.iloc[i, 4]} for i in np.arange(13,24)]



@app.route('/')
@app.route('/home')
def home():
	#return (popular_list.iloc[0][0])
	return render_template('home.html', books_stats=poppular_book)
	#return

@app.route("/search/<popular_books>", methods=['GET', 'POST'])
def Search():
	form = SearchForm()
	if form.validate_on_submit():
		flash(f'Recommendations for {form.search.data}','success')
		return redirect(url_for('Search',books_stats=books_stats2))
	return render_template('about.html', title='Search Book', form=form,books_stats=popular_books)

@app.route("/recs")
def about():
    return render_template('about.html', title='About',books_stats=books_stats2)

if __name__ == '__main__':

	app.run(debug=True)


# def display_popular_books():
# 	return

# key: index, value: book title
# indices = pd.Series(book_tags_cleaned.index,index=book_tags_cleaned['title'])
# input: book title 
# output: indexes of books similar to book title

# import csv
# reader = csv.reader(open("tfv_matrix.csv", "rt", encoding="utf8"), delimiter=",")
# x = list(reader)
# tfv_matrix = np.array(x).astype("float")

# cosine_sim = cosine_similarity(tfv_matrix, tfv_matrix)

# def content_based_rec(title,sig=cosine_sim):
#     book_index = indices[title]
#     sorted_scores = sorted(list(enumerate(sig[indices[book_index]])),key=lambda x: x[1],reverse=True)[1:11]
#     return book_tags_cleaned.iloc[[i[0] for i in sorted_scores],2].index.values

# content_based_books = content_based_rec(title_input)

# # content based books information 
# content_based_books_stats = [{'title': popular_list.iloc[i, 1],
# 		'isbn': popular_list.iloc[i, 2],
# 		'author': popular_list.iloc[i, 3],
# 		'img': popular_list.iloc[i, -2],
# 		'date_published':popular_list.iloc[i, 4]} for i in content_based_books]

