from flask import Flask, render_template, request, url_for, flash, redirect
#from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField,SelectField
from wtforms.validators import DataRequired, AnyOf
# from models import Book



app = Flask(__name__,template_folder='templates')
app.config['SECRET_KEY'] = 'a47cf59c318546f2c15c12d79936f384'
#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
#db = SQLAlchemy(app)

book_tags_cleaned = pd.read_csv(r'../data/processed/book_tags_cleaned.csv')
popular_list = pd.read_csv(r'../data/processed/popular_list.csv')
books_list = pd.read_csv(r'../data/processed/books_list.csv')
# indexes of popular books
recommended_books = np.arange(0,12)
books_stats = [{'title': popular_list.iloc[i, 1],
		'isbn': popular_list.iloc[i, 2],
		'author': popular_list.iloc[i, 3],
		'img': popular_list.iloc[i, -2],
		'date_published':popular_list.iloc[i, 4]} for i in recommended_books]
books_stats2 = [{'title': popular_list.iloc[i, 1],
		'isbn': popular_list.iloc[i, 2],
		'author': popular_list.iloc[i, 3],
		'img': popular_list.iloc[i, -2],
		'date_published':popular_list.iloc[i, 4]} for i in np.arange(13,24)]


@app.route('/',methods=['GET','POST'])
@app.route('/home',methods=['GET','POST'])
def home():
	#return (popular_list.iloc[0][0])
	form = SearchForm()
	if form.validate_on_submit():
		flash(f'Recommendations for {form.search.data}','success')
		return redirect(url_for('Search',book_reference=form.search.data))
	return render_template(r'home.html',form=form, books_stats=books_stats)
	#return

@app.route("/search/<book_reference>", methods=['GET', 'POST'])
def Search(book_reference):
	form=SearchForm()
	content_based = content_based_rec(book_reference)
	content_based_recs = [{'title': books_list.loc[books_list['id'] == i, 'original_title'].values[0],
	'isbn': books_list.loc[books_list['id'] == i, 'isbn'].values[0],
	'author': books_list.loc[books_list['id'] == i, 'authors'].values[0],
	'img': books_list.loc[books_list['id'] == i, 'image_url'].values[0],
	'date_published':books_list.loc[books_list['id'] == i, 'original_publication_year'].values[0]} for i in content_based]
	if form.validate_on_submit():
		flash(f'Recommendations for {form.search.data}','success')
		return redirect(url_for('Search',book_reference=form.search.data))
	return render_template(r'about.html', form=form,books_stats=content_based_recs)


# key: index, value: book title
indices = pd.Series(book_tags_cleaned['book_id'].values,index=book_tags_cleaned['title'])
# input: book title 
# output: indexes of books similar to book title

import csv
reader = csv.reader(open(r"../data/tfv_matrix.csv", "rt", encoding="utf8"))
x = list(reader)
tfv_matrix = np.array(x)

cosine_sim = cosine_similarity(tfv_matrix, tfv_matrix)
book_tags_cleaned_indices_changed = book_tags_cleaned.set_index('book_id').sort_values('book_id')

def content_based_rec(title,sig=cosine_sim):
    book_index = indices[title]
    sorted_scores = sorted(list(enumerate(sig[book_index])),key=lambda x: x[1],reverse=True)[0:11]
    return book_tags_cleaned_indices_changed.iloc[[i[0] for i in sorted_scores],1].index.values

# content_based_books = content_based_rec(title_input)

# # content based books information 
# content_based_books_stats = [{'title': popular_list.iloc[i, 1],
# 		'isbn': popular_list.iloc[i, 2],
# 		'author': popular_list.iloc[i, 3],
# 		'img': popular_list.iloc[i, -2],
# 		'date_published':popular_list.iloc[i, 4]} for i in content_based_books]

class SearchForm(FlaskForm):
	search = StringField('Search Book',validators=[DataRequired(),AnyOf(book_tags_cleaned['title'].values.tolist(),message='Sorry! We don\'t have that book!')])
	submit = SubmitField('Enter')

@app.route("/recs")
def about():
    return render_template(r'about.html', title='About',books_stats=content_based_books_stats)

if __name__ == '__main__':

	app.run(debug=True)