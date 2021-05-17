from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, AnyOf

class SearchForm(FlaskForm):
	search = StringField('Search Book',validators=[DataRequired(),AnyOf])
	submit = SubmitField('Enter')