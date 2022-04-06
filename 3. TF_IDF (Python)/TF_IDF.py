import  pandas as pd
import numpy as np
import re
import math
import nltk

# Uncomment the following two lines for the first time to download the stopwords and lemmatizer
# nltk.download('stopwords')
# nltk.download('wordnet')

# Load the 'rating.csv' and 'book.csv' using panda
ratings = pd.read_csv('input/rating.csv', names=['user_id', 'book_id', 'rating'], skiprows=1, sep='\t')
books = pd.read_csv('input/book.csv', names=['book_id', 'authors', 'year', 'title', 'language'], skiprows=1, sep='\t')

# make all the text data lowercase
books['authors'] = books['authors'].str.lower();
books['title'] = books['title'].str.lower();
books['language'] = books['language'].str.lower();

# remove unnecessary chars and punctuations
books['authors'].replace('[^A-Za-z0-9,]+', '', regex=True, inplace = True)
books['title'].replace('[^A-Za-z0-9 ]+', '', regex=True, inplace = True)

# split title to get all the words of a books
# split authors to get all the authors of a book
books['authors'] = books['authors'].str.split(',');
books['title'] = books['title'].str.split(' ');

# remove the stopwords and duplicates and then lemmatize the words of the books
stop_words = nltk.corpus.stopwords.words("english")
books['title'] = books['title'].dropna().apply(lambda x: [item for item in x if item not in stop_words])
books['title'] = books['title'].dropna().apply(lambda x: list(set(x)))
lemmatizer = nltk.stem.WordNetLemmatizer()
books['title'] = books['title'].dropna().apply(lambda x: [lemmatizer.lemmatize(item) for item in x])

# calculate DF (Document Frequency) // number of occurance
# (i) for every word of the title column
# (ii) for every author of the authors column
# (iii) for every language of the language column
# (iv) for every year of the year column
DF_title = {}
DF_authors = {}
DF_language = {}
DF_year = {}
for index, row in books.dropna().iterrows():
    for w in row['title']:
        try:
            DF_title[w].add(row['book_id'])
        except:
            DF_title[w] = {row['book_id']}

    for w in row['authors']:
        try:
            DF_authors[w].add(row['book_id'])
        except:
            DF_authors[w] = {row['book_id']}

    try:
        DF_language[row['language']].add(row['book_id'])
    except:
        DF_language[row['language']] = {row['book_id']}

    try:
        DF_year[row['year']].add(row['book_id'])
    except:
        DF_year[row['year']] = {row['book_id']}

# calculate IDF (Inverse Document Frequency) // formula -> (log_10 (number of books / DF value))
# (i) for every word of the title column // same for all the words of a book because of binary representation
# (ii) for every author of the authors column  // same for all the authors of a book because of binary representation
# (iii) for every language of the language column
# (iv) for every year of the year column
IDF_title = {}
IDF_authors = {}
IDF_language = {}
IDF_year = {}
num_books = len(books.index)
for key, value in DF_title.items():
    IDF_title[key] = math.log(num_books / (len(value) + 1), 10)
for key, value in DF_authors.items():
    IDF_authors[key] = math.log(num_books / (len(value) + 1), 10)
for key, value in DF_language.items():
    IDF_language[key] = math.log(num_books / (len(value) + 1), 10)
for key, value in DF_year.items():
    IDF_year[key] = math.log(num_books / (len(value) + 1), 10)

# calculate normalized TF (Term Frequency) // Because it is binary representation, normalized TF Formula
# (i) for title -> square_root(1 / number of total words in a title in a book)
# (ii) for author -> square_root(1 / number of total authors in a book)
# (iii) for language -> 1.0 [because only one language in a book]
# (iv) for year -> 1.0 [because only one year in a book]
TF_title = {}
TF_authors = {}
TF_language = {}
TF_year = {}
for index, row in books.dropna().iterrows():
    TF_title[row['book_id']] = 1 / math.sqrt(len(row['title'])) if len(row['title'])>0 else 0
    TF_authors[row['book_id']] = 1 / math.sqrt(len(row['authors']))
    TF_language[row['book_id']] = 1.0
    TF_year[row['book_id']] = 1.0

# create user profile // formula -> (TF value * User rating)
# (i) for every word of the title column
# (ii) for every author of the authors column
# (iii) for every language of the language column
# (iv) for every year of the year column
UA_title = {}
UA_authors = {}
UA_language = {}
UA_year = {}
for index, row in ratings.iterrows():
    book = books.loc[books['book_id'] == row['book_id']].iloc[0]

    if row['user_id'] not in UA_title:
        UA_title[row['user_id']] = {}
    if row['user_id'] not in UA_authors:
        UA_authors[row['user_id']] = {}
    if row['user_id'] not in UA_language:
        UA_language[row['user_id']] = {}
    if row['user_id'] not in UA_year:
        UA_year[row['user_id']] = {}

    if row['book_id'] in TF_title:
        for w in book['title']:
            if w not in UA_title[row['user_id']]:
                UA_title[row['user_id']][w] = TF_title[row['book_id']]*row['rating']
            else:
                UA_title[row['user_id']][w] += TF_title[row['book_id']] * row['rating']

    if row['book_id'] in TF_authors:
        for w in book['authors']:
            if w not in UA_authors[row['user_id']]:
                UA_authors[row['user_id']][w] = TF_authors[row['book_id']]*row['rating']
            else:
                UA_authors[row['user_id']][w] += TF_authors[row['book_id']] * row['rating']

    if row['book_id'] in TF_language:
        if book['language'] not in UA_language[row['user_id']]:
            UA_language[row['user_id']][book['language']] = TF_language[row['book_id']] * row['rating']
        else:
            UA_language[row['user_id']][book['language']] += TF_language[row['book_id']] * row['rating']

    if row['book_id'] in TF_year:
        if book['year'] not in UA_year[row['user_id']]:
            UA_year[row['user_id']][book['year']] = TF_year[row['book_id']] * row['rating']
        else:
            UA_year[row['user_id']][book['year']] += TF_year[row['book_id']] * row['rating']


############ Generating Output files ###############

# write IDF value in 4 files
IDF_title_output = open("output/IDF_title.csv", "w")
IDF_title_output.write("word\tIDF_val\n")
for key,val in IDF_title.items():
    line = str(key) + "\t" + str(val) + "\n"
    IDF_title_output.write(line)
IDF_title_output.close()

IDF_authors_output = open("output/IDF_authors.csv", "w")
IDF_authors_output.write("author\tIDF_val\n")
for key,val in IDF_authors.items():
    line = str(key) + "\t" + str(val) + "\n"
    IDF_authors_output.write(line)
IDF_authors_output.close()

IDF_language_output = open("output/IDF_language.csv", "w")
IDF_language_output.write("language\tIDF_val\n")
for key,val in IDF_language.items():
    line = str(key) + "\t" + str(val) + "\n"
    IDF_language_output.write(line)
IDF_language_output.close()

IDF_year_output = open("output/IDF_year.csv", "w")
IDF_year_output.write("year\tIDF_val\n")
for key,val in IDF_year.items():
    line = str(key) + "\t" + str(val) + "\n"
    IDF_year_output.write(line)
IDF_year_output.close()


# write TF values in 4 files
TF_title_output = open("output/TF_title.csv", "w")
TF_title_output.write("book_id\tword\tTF_title_val\n")
for index, row in books.dropna().iterrows():
    for word in row['title']:
        line = str(row['book_id']) + "\t" + word + "\t" + str(TF_title[row['book_id']]) + "\n"
        TF_title_output.write(line)
TF_title_output.close()

TF_authors_output = open("output/TF_authors.csv", "w")
TF_authors_output.write("book_id\tauthor\tTF_author_val\n")
for index, row in books.dropna().iterrows():
    for author in row['authors']:
        line = str(row['book_id']) + "\t" + author + "\t" + str(TF_authors[row['book_id']]) + "\n"
        TF_authors_output.write(line)
TF_authors_output.close()

TF_language_output = open("output/TF_language.csv", "w")
TF_language_output.write("book_id\tlanguage\tTF_language_val\n")
for index, row in books.dropna().iterrows():
    line = str(row['book_id']) + "\t" + row['language'] + "\t" + str(TF_language[row['book_id']]) + "\n"
    TF_language_output.write(line)
TF_language_output.close()

TF_year_output = open("output/TF_year.csv", "w")
TF_year_output.write("book_id\tyear\tTF_year_val\n")
for index, row in books.dropna().iterrows():
    line = str(row['book_id']) + "\t" + str(row['year']) + "\t" + str(TF_year[row['book_id']]) + "\n"
    TF_year_output.write(line)
TF_year_output.close()


# write user profile in 4 files
UA_title_output = open("output/UA_title.csv", "w")
UA_title_output.write("user_id\tword\tUA_title_val\n")
for user_id in UA_title:
    for key, val in UA_title[user_id].items():
        line = str(user_id) + "\t" + str(key) + "\t" + str(val) + "\n"
        UA_title_output.write(line)
UA_title_output.close()

UA_authors_output = open("output/UA_authors.csv", "w")
UA_authors_output.write("user_id\tauthor\tUA_author_val\n")
for user_id in UA_authors:
    for key, val in UA_authors[user_id].items():
        line = str(user_id) + "\t" + str(key) + "\t" + str(val) + "\n"
        UA_authors_output.write(line)
UA_authors_output.close()

UA_language_output = open("output/UA_language.csv", "w")
UA_language_output.write("user_id\tlanguage\tUA_language_val\n")
for user_id in UA_language:
    for key, val in UA_language[user_id].items():
        line = str(user_id) + "\t" + str(key) + "\t" + str(val) + "\n"
        UA_language_output.write(line)
UA_language_output.close()

UA_year_output = open("output/UA_year.csv", "w")
UA_year_output.write("user_id\tyear\tUA_year_val\n")
for user_id in UA_year:
    for key, val in UA_year[user_id].items():
        line = str(user_id) + "\t" + str(key) + "\t" + str(val) + "\n"
        UA_year_output.write(line)
UA_year_output.close()
