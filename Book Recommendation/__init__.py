from flask import Flask, render_template, request, url_for, flash, redirect
from forms import SearchForm
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a47cf59c318546f2c15c12d79936f384'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

from book_rec_app import routes

