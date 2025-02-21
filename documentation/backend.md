# Backend: Technical Instructions

This document describes setting up and using the backend for Movie Book Recommender.

## Set-up virtual machine in cPouta

To create a virtual machine in cPouta, instructions from [CSC](https://docs.csc.fi/cloud/pouta/launch-vm-from-web-gui/#preparatory-steps) were followed. Some highlights: 
1. **CSC project.**
    The correct CSC project was selected from [MyCSC](https://my.csc.fi/) (CSC userID and password are needed to access the project information).
2. **SSH keys.** 
    SSH keys were set up for the project. Public key was generated in cPouta. Private key was extracted and stored with the help of [puttygen](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html).
3. **Firewalls and security groups.** 
    The movie book recommender uses five cPouta security groups: default and ssh (available by default in cPouta) as well as postgres (new one created to open a port to access the database), and watchtower and docker (new ones created to open ports to deploy frontend to cPouta). Details on the security groups are available for project members through [cPouta](https://pouta.csc.fi/).
4. **Launching a virtual machine.** 
    In cPouta, an instance called moviebook_test was launched. Ubuntu-22.04 was selected as the image (provided by cPouta) and standard.large was selected as the flavor.
5. **IP address.** 
    An IP address was created for the virtual machine. It is **128.214.253.51**.

## Connect to the virtual machine

To connect to the virtual machine, instructions from [CSC](https://docs.csc.fi/cloud/pouta/connecting-to-vm/) were followed. Some highlights:
1. **Keypair-based SSH connection via Putty**
    Connection to the virtual machine is based on SSH. Connection was created with the help of Putty. It is important to note that unlike stated in the CSC instructions, the connect to the virtual machine needs to to use the following user name: **ubuntu**.

    This opens connection to the virtual machine command line.

2. **Other methods to connect to the virtual machine**
    [WinSCP](https://winscp.net/) can be used to easily transfer files between local and remote machines. It uses connections set up in Putty.

3. **Upgrading and updating the virtual machine**
    Before using the virtual machine, the operating system was updated and upgraded. Updates require restarting the system.

    ```
    sudo apt update
    sudo apt upgrade
    sudo shutdown -r now
    ```

## Install, set-up and get access to a new Postgres database

* Portgres database was **installed** with the following command: 
    ```
    sudo apt install postgresql postgresql-contrib
    ```

* To get the database operational, the following **configuration files** were edited: 
    * /etc/postgresql/14/main/pg_hba.conf
    * /etc/postgresql/14/main/postgresql.conf

* Thereafter, the database was **restarted**
    ```
    sudo service postgresql restart
    ```

* Super user for the database is *postgres* by default. As the superuser, a **database** called *mvbkdb* and **user** *moviebook* (can create and populate tables as well as make queries) were **created**. 
    ```
    CREATE DATABASE mvbkdb;
    CREATE USER moviebook CREATEDB LOGIN PASSWORD password;
    ```

* Superuser can, if needed, **change the password** for *moviebook* with the following command.
    ```
    ALTER USER moviebook WITH ENCRYPTED PASSWORD newpassword;
    ```

* To access the database as *moviebook* the following command is used. Thereafter the *moviebook's* password needs to be input.
    ```
    psql -d mvbkdb -U moviebook
    ```

* The following actions were taken as user *moviebook*. Database *mvbkdb* can now be accessed from the virtual machine's Ubuntu command line. 

## Create tables in the database

* **Tables** for the database were created with the script [create_db.sql](../documentation/create_db.sql). 

* It was run with the following command:
    ```
    psql -U moviebook -d mvbkdb -a -f create_db.sql
    ```

## Download and parse data

* Data was **downloaded** straight to the virtual machine.
    ```
    wget https://files.grouplens.org/datasets/tag-genome-2021/genome_2021.zip
    wget https://files.grouplens.org/datasets/book-genome/book-genome.zip
    ```

* To **unzip** a new program was installed... 
    ```
    sudo apt install unzip

    ```
* ... and the files were unzipped with the following commands
    ```
    unzip book-genome.zip
    unzip genome_2021.zip
    ```

## Populate the database with raw data

* Due to the large size of the files, population of the database was done using program called `jq`
    ```
    sudo apt install jq
    ```

* jq is able to read json files and **convert** them to csv files with a command like the following
    ```
    cat ./movie_dataset_public_final/raw/tags.json | jq -r '. | join("|")' > tags.csv
    ```

* The resulting csv file can be imported into the database. For example:
    ```
    psql -U moviebook -d mvbkdb -c "\copy mv_tags FROM 'tags.csv' delimiter '|' csv"
    ```

* The script used to populate the database is available in [csc_json_to_csv_to_psql.sh](../documentation/csc_json_to_csv_to_psql.sh)
* Please note that to make a script executable on Linux, the following command is needed `chmod +x filename.sh`.

## Write backend APIs to serve data from database

* The **APIs** were written using python, Flask, and SQLAlchemy. The following commands were used to install the needed packages:
    ```
    sudo apt-get install python3-pip libpq-dev python3-flask
    pip install sqlalchemy psycopg2 flask_sqlalchemy
    ```

* Code for the backend APIs is available for [movies](../main/movies.py) and [books](../main/routes_books.py).

* To establish connection with the database, the following command needs to be run on the command line:
    ```
    export DATABASE_URL="postgresql://user:password@128.214.253.51:5432/mvbkdb"
    ```

* Backend APIs can be run from command line on the virtual machine with the following command:
    ```
    flask run --host=0.0.0.0 --port=3000
    ```

## Examples of API endpoints available

* Output from the backend API is available via the address [http://128.214.253.51:3000/](http://128.214.253.51:3000/).

* Data can be accessed by using the routes created for [movies](../main/routes_movies.py) and [books](../main/routes_books.py)
* For example the following data is available for use in the front end:

1. All information available for **one movie** 
    * Data is provided via the route `@app.route('/dbgetonemoviedata', methods = ['GET'])`
    * You will need to specify the movie based on the movieid in the tmbd_movie_data_full table, for example movie *Heat* has movieid 6.
    * Results are available in JSON form from address [http://128.214.253.51:3000/dbgetgivenmoviedata?movieid=6](http://128.214.253.51:3000/dbgetgivenmoviedata?movieid=6).
    * Output looks as follows (note: this page is only done for debugging and illustrative purposes. In reality, front end uses just the routes and addresses stated above)
    ![Heat](one_movie.jpg)

2. All information available for **top 10 movies by revenue for a given year**
    * Data is provided via the route `@app.route('/dbgettop10moviesbyyear', methods = ['GET']`
    * You will need to specify the release year of the movie.
    * Results are available in JSON form from address [http://128.214.253.51:3000/dbgettop10moviesbyyear?year=2010](http://128.214.253.51:3000/dbgettop10moviesbyyear?year=2010).

3. All information available for various sorting methods
    * **Newest 10 movies**: Data is provided via the route `@app.route('/dbgettop10newestpublishedmovies', methods = ['GET'])`
    * **Oldest 10 movies**: Data is provided via the route `@app.route('/dbgettop10oldestmovies', methods = ['GET'])`
    * **Highest rated 10 movies**: Data is provided via the route `@app.route('/dbgettop10highestratedmovies', methods = ['GET'])`
    * **Search based on name**: Also, movies can be searched based on name via the route `@app.route('/dbsearchmoviesbyname', methods = ['GET'])`

4. APIs for books follow the same logic. 
    * **Newest 10 books**: Data is provided via the route `@app.route('/dbgettop10newestbooks', methods = ['GET'])`
    * **All information for one book**: Data is provided via the route `@app.route('/dbgetgivenbookdata', methods = ['GET'])`, e.g. [http://128.214.253.51:3000/dbgetgivenbookdata?bookid=44752519](http://128.214.253.51:3000/dbgetgivenbookdata?bookid=44752519)
    * **Search based on name**: Books can be searched based on name via the route `@app.route('/dbsearchbooksbyname', methods = ['GET'])`, e.g. [http://128.214.253.51:3000/dbsearchbooksbyname?input=Sookie%20stackhouse](http://128.214.253.51:3000/dbsearchbooksbyname?input=Sookie%20stackhouse)

* The JSON output for movies provides fields such as *posterpath* and *backdroppaths*. These are actually just the file names, but can be joined together with the string https://image.tmdb.org/t/p/original in order to build the full url. For example, https://image.tmdb.org/t/p/original/obpPQskaVpSiC9RcJRB6iWDTCXS.jpg

## Populate the database with data on recommendations

* Core functionality in the application lies in an algorithm that calculates recommendations for movies and books. For further details, please see a [video](https://www.youtube.com/watch?v=RD5Xz02x33U) outlining the algorithm used in the research project.

* Use of the algorithm requires additional packages:
    ```
    pip install pandas
    ```

* Research project has previously calculated for each movie and book ratings that indicate the movie's/book's characteristics. Based on this data, an algorithm calculates recommendations between movies and books for the following pairs:

    * [Movie-to-Movies](../documentation/mv_to_mvs.py)
    * [Book-to-Books](../documentation/book_to_books.py)
    * [Movie-to-Books](../documentation/movie_to_books.py)
    * [Book-to-Movies](../documentation/book_to_movies.py)

* As the results from these calculations do not change over time, the algorithms have been run and their results populated in the database. Please see related commands [here](../documentation/recommendations.sh).

## Optimize the algorithm to generate recommendations based on user preferences

* One of the core functionalities of the work is to generate recommendations for users based on their indicated preferences. The algorithm is available is in the class [Recommendations](../main/recommendations.py).
* As the algorithm has been originally developed for academic use, not for a production environment, it was modified slightly to improve memory usage and time efficiency and to make it stable in production. Primary efforts focused on limiting the tags used in calculations to the most significant ones, since the original data sets were over 300MB for movies and over 200MB for books, and hence impractical to use when calculating the user specific recommendations. 
* First, datasets for movies and books were limited to **focus only on common tags**, i.e. tags that are available for both movies and books, because these are the only tags used by the algorithm. Script to generate a list of common tags and their ids, is available in [common_tags.py](common_tags.py). They are also saved in the Postgres database in the table called [common_tags](create_db.sql).
* Second, as the movie and book tags are represented as text, which is more memory consuming, all the common tags used replaced with numeric common tag ids. Scripts to generate tag datasets that only use numeric data for [movies](movies_tagdl_common.py) and [books](books_tagdl_common.py) were developed. These first two steps reduce the movies tags from 10,551,657 lines to 4,779,395 lines, file size reducing to 112M, and the books tags from 6,814,900 lines to 4,602,635 lines, file size reducing to 129MB. These datasets are available in the Postgres database in tables *movies_tagdl_common* and *books_tagdl_common*. These actions speed up the algorithm, but do not incur any reduction in the accuracy of the recommendations. 
* These optimizations in the file formats and contents do not, however, yet improve the time and memory efficiency of the algorithm enough, for it to be stable in a production environment. Therefore, a further optimization was done: tags were limited to the ones that have **the largest impact in the calculations** for the recommendations, i.e., only tags whose score is above 0.1, were kep in datasets. This [script](mvs_bks_tagdl_common_limited.py) reduced the number of tags for to 1,184,035 and for books to 810,755, which is still on average ~120 tags per movie and ~80 tags per book, which one could presume does not impact the recommendations significantly. Resulting files and data that are used in production are available here for [movies](../datasets/movies_tagdl_common_limited.csv) and [books](../datasets/books_tagdl_common_limited.csv). 
* These efforts reduced time to generate user specific recommendations from approximately 40 seconds to 4 seconds, while the recommended movies and books remained largely the same.

## Running the backend in the cPouta server

* Application can be set to run on the cPouta server with the following command
    ```
    flask run --host=0.0.0.0 --port=3000 > log.txt 2>&1 &
    ```

* This is a valid approach for development purposes. However, it should not be used in a production environment. Instead a WSGI (Web Server Gateway Interface) Gunicorn and production level HTTP server nginx need to be installed and configured.
* First, **Gunicorn** is installed
    ```
    sudo apt-get install python3-gunicorn gunicorn3
    ```
* To help enable Gunicorn to run a project, a new [wsgi.py](../wsgi.py) file is created. 
* Now it will be possible whether Gunicorn works by running
    ```
    sudo gunicorn --workers 5 wsgi:app
    ```
* Also a system service was created so that Gunicorn is run in the background. The [configuration file](../documentation/mvbkbackend.service) is `/etc/systemd/system/mvbkbackend.service`. Configuration refers to [service_start.sh](../documentation/anonymized_service_start.sh), which is anonymized in this documentation. Currently, three workers are started.

* Then, **nginx** is installed and configured
    ```
    sudo apt-get install nginx
    ```
* The [configuration file](mvbkbackend.conf) is `/etc/nginx/sites-available/mvbkbackend.conf`.
* A particular site configuration is enable using command 
    ```
    sudo ln -s /etc/nginx/sites-available/mvbkbackend.conf /etc/nginx/sites-enabled/
    ```
* nginx can be started with the command
    ```
    sudo systemctl start nginx
    ```
* Useful instruction for installing and configuring can be found from e.g. [here](https://linuxhint.com/use-nginx-with-flask/).

## CI/CD pipeline

* The CI/CD pipeline is automated, and the steps taken to deploy to production are part of GitHub workflows:
    * [Main workflow](../.github/workflows/main.yml) is run when new code is pushed to GitHub. The script checks quality of the code, but does not deploy it to the virtual machine.
    * [Deployment workflow](../.github/workflows/deploy.yml) is run when the changes in the code are ready to be deployed to the virtual machine.

## Rebooting the back and front end

* Note! Before running system updates, it is important to make back-up of key files in the virtual machine, both as *ubuntu* and *mvbkrunner* users.
* After updating and upgrading the software, in cases of larger system updates (e.g. Linux kernel is updated), the entire virtual machine needs to be **rebooted**.

    ```
    sudo apt update
    sudo apt upgrade
    sudo reboot now
    ```

* After this, the **backend** may need to be restarted. To do this, log into the virtual machine as mvbkrunner. Follow key steps in the [deployment workflow](../.github/workflows/deploy.yml). 

* Finally, the **front end** may need to be restarted. To do this, log into the virtual machine as *ubuntu*. Follow the key steps in the publish.sh script (link to be added).
