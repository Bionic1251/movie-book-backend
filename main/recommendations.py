"""
This module implements methods to get the data and 
to calculate recommendations based on user's preferences.
"""

import pandas as pd

class Recommendations:
    """Class that generates personal recommendations for users.
       Main functions are get_movie_recommendations and
       get_book_recommendations.
    """
    def __init__(self):
        """Initializes tag resources from database.
        """
        self.data_uploaded = False
        self.get_data()
        pass

    def get_data(self):
        """ Method opens and reads files containing data for movies' and
            books' tags if the application is not run on GitHub.
            Note. Data used contains tag files that are limited to tags
            where the score is above 0.1.
        """
        self.tg_movies = pd.read_csv("./datasets/tagdl_movies_common.csv")
        self.tg_books = pd.read_csv("./datasets/tagdl_books_common.csv")
        self.feature_name = "tag"

        self.data_uploaded = True
        print("Data uploaded")

    def get_user_profile(self, tg, domain_ratings):
        """Generates a dataframe that represents user's preferences.

        Args:
            tg (dataframe): Dataframe that includes common tags.
            domain_ratings (dict): includes ratings made by user.

        Returns:
            Dataframe: Dataframe that includes the vector for the user.
        """
        df = pd.DataFrame()

        for rating in domain_ratings:
            weight = rating["rating"] - 2.5
            item = tg[tg.item_id == rating["item_id"]].copy()
            item.score = item.score * weight
            df = pd.concat([df, item])

        return df

    def get_dot_product(self, profile, tg_df):
        """Generates dot product for all items

        Args:
            profile (Dataframe): Created user profile that was created in get_user_profile.
            tg_df (Dataframe): Tag Dataframe

        Returns:
            Dataframe: Dataframe that has the dot products
        """
        tg_domain_profile = pd.merge(tg_df, profile, on=self.feature_name, how="inner")
        tg_domain_profile["dot_product"] = tg_domain_profile.score_x * tg_domain_profile.score_y
        dot_product_df = tg_domain_profile.groupby("item_id").dot_product.sum().reset_index()
        return dot_product_df

    def get_item_length_df(self, tg_df):
        """Calculates length for each movie

        Args:
            tg_df (Dataframe): The tag dataframe

        Returns:
            Dataframe: Dataframe that includes item_id and length
        """
        len_df = tg_df.copy()
        len_df["length"] = len_df.score * len_df.score
        len_df = len_df.groupby("item_id")["length"].sum().reset_index()
        len_df["length"] = len_df["length"]**(1/2)

        return len_df

    def get_vector_length(self, profile):
        """Calculates the vector length for the user

        Args:
            profile (dataframe): Dataframe with user data

        Returns:
            float: The calculated vector length.
        """

        prof_tmp = profile.copy()
        prof_tmp.score = prof_tmp.score * prof_tmp.score
        profile_vector_len = prof_tmp.score.sum()
        profile_vector_len = profile_vector_len**(1/2)

        return profile_vector_len

    def get_sim_df(self, dot_product_df, len_df, profile_vector_len):
        """Generates similarities for movies/books based on user profile

        Args:
            dot_product_df (dataframe)
            len_df (dataframe)
            profile_vector_len (float)

        Returns:
            Dataframe: Dataframe that has  the similarities for each item
        """
        sim_df = pd.merge(dot_product_df, len_df)
        sim_df["sim"] = sim_df["dot_product"] / sim_df["length"] / profile_vector_len
        return sim_df

    def get_movie_recommendations(self, ratings, amount, profile):
        """Main function to fetch recommendations for movies based on ratings.

        Args:
            ratings (dict): Dictionary that has fields movies and books
                            that have the ratings as id and value
            amount (int): The amount of results
            profile (DataFrame): profile vector of the user

        Returns:
            list: List of id's, which are the recommended movies
                  for the user. Best one is at index 0.
        """

        movie_dot_product = self.get_dot_product(profile, self.tg_movies)
        movie_len_df = self.get_item_length_df(self.tg_movies)
        profile_vector_len = self.get_vector_length(profile)
        movie_sim_df = self.get_sim_df(movie_dot_product, movie_len_df, profile_vector_len)

        # excluding rated movies
        movie_ids = []
        for element in ratings["movies"]:
            movie_ids.append(element['item_id'])
        results = movie_sim_df[~movie_sim_df.item_id.isin(movie_ids)].sort_values("sim", ascending=False, ignore_index=True).head(amount).drop(columns=["dot_product", "length", "sim"])

        results = results["item_id"].values.tolist()

        return results

    def get_book_recommendations(self, ratings, amount, profile):
        """
        Main function to fetch recommendations for books based on ratings.
        Args:
            ratings (dict): Dictionary that has fields movies and
                            books that have the ratings as id and value
            amount (int): The amount of results
            profile (DataFrame): profile vector of the user
        Returns:
            list: List of id's, which are the recommended movies for
                  the user. Best one is at index 0.
        """

        book_dot_product = self.get_dot_product(profile, self.tg_books)
        book_len_df = self.get_item_length_df(self.tg_books)
        profile_vector_len = self.get_vector_length(profile)
        book_sim_df = self.get_sim_df(book_dot_product, book_len_df, profile_vector_len)

        #excluding rated books
        book_ids = []
        for element in ratings["books"]:
            book_ids.append(element['item_id'])
        results = book_sim_df[~book_sim_df.item_id.isin(book_ids)].sort_values("sim", ascending=False, ignore_index=True).head(amount).drop(columns=["dot_product", "length", "sim"])



        results = results["item_id"].values.tolist()

        return results

    def get_all_recommendations(self, ratings, amount):
        """Function to fetch both movie and book recommendations

        Args:
            ratings (dict): Dictionary that has fields movies and books
                            that have the ratings as id and value
            amount (int): The amount of results wanted
        Returns:
            dict: Dictionary that has keys "movies" and "books"
            which includes corresponding lists of recommended item ID's.
        """

        if self.data_uploaded is False:
            print("data is not yet uploaded")
            self.get_data()
        else:
            print("data is already uploaded")

        profile = self.get_user_profile(self.tg_books, ratings["books"])
        profile = pd.concat([profile, self.get_user_profile(self.tg_movies, ratings["movies"])])
        profile = profile.groupby("tag").score.mean().reset_index()

        results = {}
        results["movies"] = self.get_movie_recommendations(ratings, amount, profile)
        results["books"] = self.get_book_recommendations(ratings, amount, profile)

        return results

recommendations = Recommendations()
