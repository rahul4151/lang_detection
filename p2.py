import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import *


def clean_data(txt):
    txt = txt.lower()
    txt = txt.replace("'","")
    txt = word_tokenize(txt)
    txt = [t for t in txt if t not in punctuation]
    txt = [t for t in txt if t not in stopwords.words("english")]
    txt = (" ").join(txt)
    return txt


#save the model  and save the vector

f = open("lg_model.pkl","rb")
model =load(f)
f.close()


f = open("lg_vector.pkl","rb")
tv = load(f)
f.close()


txt = input("Enter text: ")
ctxt = clean_data(txt)
vtxt = tv.transform([ctxt])
res  = model.predict(vtxt.toarray())
print(res[0])

