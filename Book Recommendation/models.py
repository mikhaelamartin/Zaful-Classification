from book_app import db


class Book(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	title = db.Column(db.String(20),unique=True,nullable=False)
	author = db.Column(db.String(20),nullable=False)
	image_file = db.Column(db.String(20),nullable=False,default='default.jpg')
	isbn = db.Column(db.Integer)
	date_published = db.Column(db.Integer)
	description = db.Column(db.Text,nullable=False)

	def __repr__(self):
		return f"Book('{self.title}', '{self.author}','{self.image_file}')"

