import string
import numpy as np
import pandas as pd
# import cudf
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
print('See which languages are supported:\n' + " ".join(SnowballStemmer.languages))

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer


def removePunctuation(text):
    punc_translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return text.translate(punc_translator)

# my_stop_words = text.ENGLISH_STOP_WORDS.union(["book"])
# vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=my_stop_words)
# X = vectorizer.fit_transform(["this is an apple.", "this is a book."])
# idf_values = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
#
# # printing the tfidf vectors
# print(X)
# # printing the vocabulary
# print(vectorizer.vocabulary_)

train_df = pd.read_csv('/home/ace19/dl_data/shopee-product-matching/train.csv')

train_df['title_clean'] = train_df['title'].apply(removePunctuation)
title_to_use = train_df['title_clean']

nltk.download('punkt')
nltk.download('stopwords')
print(stopwords.fileids())

# https://github.com/har07/PySastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

words = word_tokenize('READY Lemonilo Mie instant sehat kuah dan goreng')
#print(words) #
clean_words = []
for word in words:
    word = word.lower()
    if word not in stopwords.words('indonesian'): #불용어 제거
        # stemmer = SnowballStemmer('english')
        # word = stemmer.stem(word) #어간 추출
        output = stemmer.stem(word)
        print(output)
        clean_words.append(word)
#     print(clean_words)
tmp = ' '.join(clean_words)

print('Computing text embeddings...')
tfidf_vec = TfidfVectorizer(stop_words='english',
                            binary=True,
                            lowercase=True,
                            # max_df = 0.5,
                            # min_df = 2,
                            max_features=25000)
text_embeddings = tfidf_vec.fit_transform(title_to_use).toarray().astype(np.float32)
print('text embeddings shape', text_embeddings.shape)
