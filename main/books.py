"""
This module outlines the contents of all the tables with data
regarding books in the application's database.
"""

from main.extentions import db

class TableBkMetadata(db.Model):
    """This class outlines the structure of the BkMetadata table
    in the application's database, and the related methods.

    Args:
        db (object): Tables contains metadata for books.

    Returns:
        Integer, string: Returns data as integers or strings.
    """
    __tablename__ = 'bk_metadata'
    bk_metadata_item_id = db.Column('item_id', db.Integer, primary_key=True)
    bk_metadata_url = db.Column('url', db.String(1000))
    bk_metadata_title = db.Column('title', db.String(255))
    bk_metadata_authors = db.Column('authors', db.String(2000))
    bk_metadata_lang = db.Column('lang', db.String(255))
    bk_metadata_img = db.Column('img', db.String(1000))
    bk_metadata_year = db.Column('year', db.Integer)
    bk_metadata_description = db.Column('description', db.String(65535))

    def object_to_dictionary(self):
        """This method returns contents of the BkMetadata table
        in a dictionary format.

        Returns:
            Dictionary: Returns contents of the BkMetadata table.
        """
        return {
            'item_id': self.bk_metadata_item_id,
            'url': self.bk_metadata_url,
            'title': self.bk_metadata_title,
            'authors': self.bk_metadata_authors,
            'lang': self.bk_metadata_lang,
            'img': self.bk_metadata_img,
            'year': self.bk_metadata_year,
            'description': self.bk_metadata_description,
            }

class TableBkRatings(db.Model):
    """This class outlines the structure of the BkRatings table
    in the application's database, and the related methods.

    Args:
        db (object): Table contains ratings for books given by users.

    Returns:
        Integer, string: Returns data as integers.
    """
    __tablename__ = "bk_ratings"
    bk_ratings_item_id = db.Column("item_id", db.Integer, primary_key=True)
    bk_ratings_user_id = db.Column("user_id", db.Integer)
    bk_ratings_rating = db.Column("rating", db.Integer)

    def object_to_dictionary(self):
        """This method returns contents of the BkRatings table
        in a dictionary format.

        Returns:
            Dictionary: Returns contents of the BkRatings table.
        """
        return {
            "item_id": self.bk_ratings_item_id,
            "user_id": self.bk_ratings_user_id,
            "rating": self.bk_ratings_rating
        }

class TableBkSimilarBooks(db.Model):
    """This class outlines the structure of the BkSimilarBooks table
    in the application's database, and the related methods.

    Args:
        db (object): Table contains for each book the top 251 books 
        that are most similar to this book.

    Returns:
        Integer, string: Returns data from table as integers or strings.
    """
    __tablename__ = "bk_similar_books"
    bk_similar_books_item_id = db.Column('item_id', db.Integer)
    bk_similar_books_similar_item_id = db.Column('similar_item_id', db.Integer)
    bk_similar_books_similar_item_type = db.Column('similar_item_type', db.String(16))
    bk_similar_books_similarity_score = db.Column('similarity_score', db.Integer)
    bk_similar_books_row_index = db.Column('row_index', db.Integer, primary_key=True)

    def object_to_dictionary(self):
        """This method returns contents of the BkSimilarBooks table
        in a dictionary format.

        Returns:
            Dictionary: Returns contents of the BkSimilarBooks table.
        """
        return {
            'item_id': self.bk_similar_books_item_id,
            'similar_item_id': self.bk_similar_books_similar_item_id,
            'similar_item_type': self.bk_similar_books_similar_item_type,
            'similarity_score': self.bk_similar_books_similarity_score,
            'row_index': self.bk_similar_books_row_index
            }

class TableBkSimilarMovies(db.Model):
    """This class outlines the structure of the BkSimilarMovies table
    in the application's database, and the related methods.

    Args:
        db (object): Table contains for each book the top 250 movies 
        that are most similar to this book.

    Returns:
        Integer, string: Returns data from table as integers or strings.
    """
    __tablename__ = 'bk_similar_movies'
    bk_similar_movies_item_id = db.Column('item_id', db.Integer)
    bk_similar_movies_similar_item_id = db.Column('similar_item_id', db.Integer)
    bk_similar_movies_similar_item_type = db.Column('similar_item_type', db.String(16))
    bk_similar_movies_similarity_score = db.Column('similarity_score', db.Integer)
    bk_similar_movies_row_index = db.Column('row_index', db.Integer, primary_key=True)

    def object_to_dictionary(self):
        """This method returns contents of the BkSimilarMovies table
        in a dictionary format.

        Returns:
            Dictionary: Returns contents of the BkSimilarMovies table.
        """
        return {
            'item_id': self.bk_similar_movies_item_id,
            'similar_item_id': self.bk_similar_movies_similar_item_id,
            'similar_item_type': self.bk_similar_movies_similar_item_type,
            'similarity_score': self.bk_similar_movies_similarity_score,
            'row_index': self.bk_similar_movies_row_index
            }
