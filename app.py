from flask import Flask,render_template,request
import difflib
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app =  Flask(__name__)

df = pd.read_csv('data.csv')

from html.parser import HTMLParser
html_parser = HTMLParser()
df['Service'] = df['Service'].apply(lambda x: html_parser.unescape(x))

import re
df['Service']= df['Service'].apply(lambda x: x.lower())

#Extracting only words from Service
df['Service'] = df['Service'].apply(lambda x: re.sub(r'#[\W]*',' ',x))

df["Service"] = df["Service"].apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',x))

Appostophes = {"'s":"is","'re":"are"}

for i in range(len(df)):
  words =  df['Service'][i]
  for word in words:
    if word == "'s":
      word= "is"
    elif word == "'re":
      word =  "are"
  df['Service'][i] = words

documents = list(df['Service'])
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
def extract_entities(name, text):
    # Lopping over the sentences of the text
    for sent in nltk.sent_tokenize(text):
        # nltk.word_tokeize returns a list of words composing a sentence
        # nltk.pos_tag returns the position tag of words in the sentence
        # nltk.ne_chunk returns a label to each word based on this position tag when possible
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            try:
                if chunk.label() == 'PERSON':
                    for c in chunk.leaves():
                        if str(c[0].lower()) not in name:
                            name.append(str(c[0]).lower())
            except AttributeError:
                pass
    return name
## 
names = []
for doc in documents:
    names = extract_entities(names, doc)
## Update the stop words list
stop_words.update(names)


from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")
texts = [[stemmer.stem(word) for word in document.lower().split() if (word not in stop_words)]
          for document in documents]


from gensim import corpora, models, similarities
dictionary = corpora.Dictionary(texts)


import itertools
list(itertools.islice(dictionary.token2id.items(), 0, 20))

corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]


import gensim
import numpy as np

corpus_tfidf_csr = gensim.matutils.corpus2csc(corpus_tfidf)
corpus_tfidf_numpy = corpus_tfidf_csr.T.toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 
cosine_similarities = linear_kernel(corpus_tfidf_numpy, corpus_tfidf_numpy) 
results = {}
idx = 0
row = 0
count = 0
for idx, row in df.iterrows():
   similar_indices = cosine_similarities[idx].argsort()[:-100:-1] 
   similar_items = [(cosine_similarities[idx][i], df['id'][i]) for i in similar_indices] 
   results[row['id']] = similar_items[1:]


#Checking the results

def item(id):  
  return df.loc[df['id'] == id]['website'].tolist()[0].split(' - ')[0] 

def recommend(item_id, num):
    l = []
    print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
    s = item(item_id)
    l.append(s)   
    print("-------")    
    recs = results[item_id][:num]   
    for rec in recs:
       r = "" 
       print("Recommended: " + item(rec[1]) + " (score:" +      str(rec[0]) + ")")
       r = item(rec[1])
       l.append(r)
    return l

@app.route('/',methods=['GET','POST'])
def index():
	if request.method == 'GET':
		return render_template('index.html')

	if request.method== 'POST':
		cdn_idx = -1
		cdn_name = request.form['cdn_name']
		for i in range(len(df)):
			if df['website'][i] == cdn_name:
				cdn_idx= i 

		search_name = cdn_name
		if cdn_idx!=-1:
			rec_list = recommend(item_id=cdn_idx,num=5)
			return render_template('positive.html',rec_list=rec_list,search_name=search_name) 
		else:
			return render_template('negative.html')


if __name__== '__main__':
	app.run(debug=True)