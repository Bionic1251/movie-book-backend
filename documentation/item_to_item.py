import pandas as pd
import sys

tg_books = pd.read_csv(sys.argv[1])
tg_movies = pd.read_csv(sys.argv[2])

book_tags = set(tg_books.tag.unique())
movie_tags = set(tg_movies.tag.unique())

# limiting data
common_tags = book_tags.intersection(movie_tags)
tg_movies_limited = tg_movies[tg_movies.tag.isin(common_tags)].copy()
tg_books_limited = tg_books[tg_books.tag.isin(common_tags)].copy()

# adding fields
tg_movies_limited["item_id_unmarked"] = tg_movies_limited.item_id
tg_books_limited["item_id_unmarked"] = tg_books_limited.item_id
tg_movies_limited["item_id"] = (tg_movies_limited.item_id.apply(str) + "m")
tg_books_limited["item_id"] = (tg_books_limited.item_id.apply(str) + "b")
tg_movies_limited["type"] = "movie"
tg_books_limited["type"] = "book"

tg_limited = pd.concat([tg_movies_limited, tg_books_limited])

def get_vector_length(target_item):
    item_tmp = target_item.copy()
    item_tmp.score = item_tmp.score * item_tmp.score
    item_vector_len = item_tmp.score.sum()
    item_vector_len = item_vector_len**(1/2)
    return item_vector_len

def get_dot_product(target_item, tg_df):
    tg_domain_target_item = pd.merge(tg_df, target_item, on='tag', how='inner')
    tg_domain_target_item['dot_product'] = tg_domain_target_item.score_x * tg_domain_target_item.score_y
    dot_product_df = tg_domain_target_item.groupby('item_id_x').dot_product.sum().reset_index()
    return dot_product_df

def get_item_length_df(tg_df):
    len_df = tg_df.copy()
    len_df["length"] = len_df.score * len_df.score
    len_df = len_df.groupby("item_id")["length"].sum().reset_index()
    len_df["length"] = len_df["length"]**(1/2)
    return len_df

def get_sim_df(dot_product_df, len_df, profile_vector_len):
    sim_df = pd.merge(dot_product_df, len_df, left_on="item_id_x", right_on="item_id")
    sim_df["sim"] = sim_df["dot_product"] / sim_df["length"] / profile_vector_len
    return sim_df

def get_type(field):
    if field[-1] == "m":
        return "movie"
    else:
        return "book"

def parse_id(field):
    return field[:-1]

item_len_df = get_item_length_df(tg_limited)
ids = tg_limited.item_id.unique()

for i in ids:
    print(i)
    target_item = tg_limited[tg_limited.item_id == i].copy()

    target_item_len = get_vector_length(target_item)
    dot_product = get_dot_product(target_item, tg_limited)
    related_items_sim_df = get_sim_df(dot_product, item_len_df, target_item_len)
    
    related_items_sim_df = related_items_sim_df.drop(columns=["item_id_x"])
    related_items_sim_df = related_items_sim_df.rename(columns={"item_id": "item_id_marked1", "length": "length1"})
    related_items_sim_df["item_id_marked2"] = i
    related_items_sim_df["length2"] = target_item_len
    
    #restoring types and ids
    related_items_sim_df["type1"] = related_items_sim_df.item_id_marked1.apply(get_type)
    related_items_sim_df["type2"] = related_items_sim_df.item_id_marked2.apply(get_type)
    related_items_sim_df["item_id1"] = related_items_sim_df.item_id_marked1.apply(parse_id).astype(int)
    related_items_sim_df["item_id2"] = related_items_sim_df.item_id_marked2.apply(parse_id).astype(int)

    #repeating records
    related_items_sim_df_copy = related_items_sim_df.copy()
    related_items_sim_df_copy = related_items_sim_df_copy.rename(
        columns={"item_id_marked1": "item_id_marked2", "item_id1": "item_id2", "length1": "length2", "type1": "type2",
                 "item_id_marked2": "item_id_marked1", "item_id2": "item_id1", "length2": "length1", "type2": "type1"})
    item_sim_df = pd.concat([related_items_sim_df, related_items_sim_df_copy])

    item_sim_df[['item_id_marked1', 'item_id1', 'type1', 'length1', 'item_id_marked2', 'item_id2', 'type2', 'length2',
                 'dot_product', 'sim']].to_csv("item_to_item_sim.csv", mode='a', index=False, header=False)
