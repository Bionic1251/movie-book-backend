# These commands insert basic data related to movies and books to database.
cat ./movie_dataset_public_final/raw/metadata.json | jq -r '. | join("^")' > metadata.csv
cat ./movie_dataset_public_final/raw/metadata_updated.json | jq -r '. | join("^")' > metadata_updated.csv
cat ./movie_dataset_public_final/raw/ratings.json | jq -r '. | join("|")' > ratings.csv
cat ./movie_dataset_public_final/raw/reviews.json | jq -r '. | [.item_id, (.txt | gsub("\r|\n|\r\n|\n\r|\u0085|\t"; " "))] | join("\t")' > reviews.csv
cat ./movie_dataset_public_final/raw/reviews.json | jq '. | join("|")' > reviews.csv
cat ./movie_dataset_public_final/raw/survey_answers.json | jq -r '. | join("|")' > survey_answers.csv
cat ./movie_dataset_public_final/raw/tag_count.json | jq -r '. | join("|")' > tag_count.csv
cat ./movie_dataset_public_final/raw/tags.json | jq -r '. | join("|")' > tags.csv

#export PGPASSWORD=password

psql -U moviebook -d mvbkdb -c "\copy mv_metadata_updated FROM 'metadata_updated.csv' delimiter '^' quote E'\b' csv"
psql -U moviebook -d mvbkdb -c "\copy mv_ratings FROM 'ratings.csv' delimiter '|' csv"
psql -U moviebook -d mvbkdb -c "\copy mv_metadata FROM 'metadata.csv' delimiter '^' quote E'\b' csv"
psql -U moviebook -d mvbkdb -c "\copy mv_reviews FROM 'reviews.csv' delimiter E'\t' csv"
psql -U moviebook -d mvbkdb -c "\copy mv_reviews FROM 'reviews.csv' delimiter '|' csv"
psql -U moviebook -d mvbkdb -c "\copy mv_survey_answers FROM 'survey_answers.csv' delimiter '|' csv"
psql -U moviebook -d mvbkdb -c "\copy mv_tag_count FROM 'tag_count.csv' delimiter '|' csv"
psql -U moviebook -d mvbkdb -c "\copy mv_tags FROM 'tags.csv' delimiter '|' csv"
psql -U moviebook -d mvbkdb -c "\copy mv_tagdl FROM './movie_dataset_public_final/scores/tagdl.csv' delimiter ',' csv header"

cat ./book_dataset/raw/metadata.json | jq '. | join("^")' > bk_metadata.csv
sed 's/^"//;s/"$//' bk_metadata.csv > bk_metadata2.csv
cat ./book_dataset/raw/ratings.json | jq -r '. | join("|")' > bk_ratings.csv
cat ./book_dataset/raw/reviews.json | jq '. | join("\t")' > bk_reviews1.csv
sed 's/^"//;s/"$//;s/\\"//g' bk_reviews1.csv > bk_reviews2.csv
sed -E 's/^([0-9]+)\\t/\1\t/' bk_reviews2.csv > bk_reviews3.csv
cat ./book_dataset/raw/survey_answers.json | jq -r '. | join("|")' > bk_survey_answers.csv
cat ./book_dataset/raw/tag_count.json | jq -r '. | join("|")' > bk_tag_count.csv
cat ./book_dataset/raw/tags.json | jq -r '. | join("|")' > bk_tags.csv

psql -U moviebook -d mvbkdb -c "\copy bk_metadata FROM 'bk_metadata2.csv' delimiter '^' quote E'\b' csv"
psql -U moviebook -d mvbkdb -c "\copy bk_ratings FROM 'bk_ratings.csv' delimiter '|' csv"
psql -U moviebook -d mvbkdb -c "\copy bk_reviews FROM 'bk_reviews3.csv' delimiter E'\t' csv"
psql -U moviebook -d mvbkdb -c "\copy bk_survey_answers FROM 'bk_survey_answers.csv' delimiter '|' csv"
psql -U moviebook -d mvbkdb -c "\copy bk_tag_count FROM 'bk_tag_count.csv' delimiter '|' csv"
psql -U moviebook -d mvbkdb -c "\copy bk_tags FROM 'bk_tags.csv' delimiter '|' csv"
psql -U moviebook -d mvbkdb -c "\copy bk_tagdl FROM './book_dataset/scores/tagdl.csv' delimiter ',' csv header"

# Note! In this file there were a limited number of errors, which were corrected manually
cat ./movie_tmdb_data_full.json | jq -r '.[] | join("^")' > movie_tmdb_data_full.csv
psql -U moviebook -d mvbkdb -c "\copy movie_tmdb_data_full FROM 'movie_tmdb_data_full2a.csv' delimiter '^' quote E'\b' csv"