import re
import os
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models, similarities

import itertools
import json
from collections import Counter
from itertools import chain
import math
import numpy as np

from flask import Flask,request
from flask import render_template
app = Flask(__name__)

#-----------------------------------------------------------------------------

# Implementing LESK algorithm from a IEEE paper. 

def lesk(context_sentence, ambiguous_word, pos=None, stem=True, hyperhypo=True):
    ps = PorterStemmer()
    max_overlaps = 0; lesk_sense = None
    context_sentence = context_sentence.split()
    for ss in wn.synsets(ambiguous_word):
        # If POS is specified.
        if pos and ss.pos is not pos:
            continue

        lesk_dictionary = []

        # Includes definition.
        lesk_dictionary+= ss.definition().split()
        # Includes lemma_names.

        lesk_dictionary+= ss.lemma_names()

        # Optional: includes lemma_names of hypernyms and hyponyms.
        if hyperhypo == True:
            lesk_dictionary+= list(chain(*[i.lemma_names() for i in ss.hypernyms()+ss.hyponyms()]))       

        if stem == True: # Matching exact words causes sparsity, so lets match stems.
            lesk_dictionary = [ps.stem(i) for i in lesk_dictionary]
            context_sentence = [ps.stem(i) for i in context_sentence] 

        overlaps = set(lesk_dictionary).intersection(context_sentence)

        if len(overlaps) > max_overlaps:
            lesk_sense = ss
            max_overlaps = len(overlaps)
    return lesk_sense


def common(input1,input2):
    commonlist=[]
    for each in input1:
        if each not in commonlist:
            commonlist.append(each)
    for each in input2:
        if each not in commonlist:
            commonlist.append(each)
    return commonlist

def common1(input1,input2):
    commonword=[]
    commonsim=[]
    for each in input1:
        if type(each)==str:
            if each not in commonword:
                commonword.append(each)
        else:
            if each not in commonsim:
                commonsim.append(each)
    for each in input2:
        if type(each)==str:
            if each not in commonword:
                commonword.append(each)
        else:
            if each not in commonsim:
                commonsim.append(each)
    return commonword+commonsim


def nps(sentence):
    senses=[]
    wordlist=sentence.split(" ")
    for eachword in wordlist:
        x=lesk(sentence,eachword)
        if x is not None:
            senses.append(x)
        else:
            senses.append(eachword);
    return senses


def similarity_score(input_array,word):
    max_similarity=0
    for each in input_array:
        simili=word_similarity(each,word)
        if simili>max_similarity:
            max_similarity=simili
    return max_similarity

def length(word1,word2):
    return word1.shortest_path_distance(word2,simulate_root=True)
    

def word_similarity(word1,word2):
    leng=length(word1,word2)
    need_root = word1._needs_root()
    subsumers = word1.lowest_common_hypernyms(word2, simulate_root=True and need_root)
    if word1==word2:
        return 1
    if len(subsumers)==0:
        depth=word1.max_depth()
    else:
        subsumer = subsumers[0]
        depth = subsumer.max_depth() + 1 
    return math.exp(-0.2*leng)*((math.exp(0.45*depth)-math.exp(-0.45*depth))/(math.exp(0.45*depth)+math.exp(-0.45*depth)))

def cosine_similarity(v1,v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)
    
def lsv(common_nps,input_nps):
    lsvector=[]
    
    input_nps_syn=[]
    input_nps_word=[]
    
    for each in input_nps:
        if type(each)==str:
            input_nps_word.append(each)
        else:
            input_nps_syn.append(each)
    
    
    for each in common_nps:
        if type(each)==str:
            if each in input_nps_word:
                lsvector.append(1)
            else:
                lsvector.append(0)
        else:
            if each in input_nps_syn:
                lsvector.append(1)
            else:
                lsvector.append(similarity_score(input_nps_syn,each))
    return lsvector
 
 
def word_order(common_nps,input_nps):
    wovector=[]
     
    for each in common_nps:
        maxi=0
        i=1
        j=1
        for every in input_nps:
            i+=1
            sim=word_similarity(each,every)
            print(each,"  ",every,"  ",sim)
            if sim>maxi:
                maxi=sim
                j=i
        wovector.append(j)
    return wovector


def word_order1(common_nps,input_nps):
    wovector=[]
     
    for each in common_nps:
        maxi=0
        i=1
        j=1
        for every in input_nps:
            if type(each)==str and type(every)==str:
                i+=1
                if each==every:
                    maxi=1
                    j=i       
            elif type(each)!=str and type(every)!=str:
                i+=1
                sim=word_similarity(each,every)
                print(each,"  ",every,"  ",sim)
                if sim>maxi:
                    maxi=sim
                    j=i
            wovector.append(j)
    return wovector



                    
def word_order_simili(sum_array,sub_array):
    sumsub,sumsum=0,0
    for each in range(len(sum_array)):
        sumsum+=sum_array[each]*sum_array[each]
    den=math.sqrt(sumsum)
    for each in range(len(sub_array)):
        sumsub+=sub_array[each]*sub_array[each]
    num=math.sqrt(sumsub)   
    return num/den





#-------------------------------------------------------------------------------


def document_to_wordlist( review, remove_stopwords=False ):
	'''
		Takes a string and converts it to wordlist(list)
	'''
	review_text = BeautifulSoup(review).get_text()
	review_text = re.sub("[^a-zA-Z]"," ", review_text)
	words = review_text.lower().split()
	if remove_stopwords:
		stops = set(stopwords.words("english"))
		words = [w for w in words if not w in stops]
	return(words)

    
def document_to_sentences( review, tokenizer, remove_stopwords=False ):
	'''
		Takes a document and inputs it to a list of lists.
	'''
	raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
	sentences = []
	for raw_sentence in raw_sentences:
	    if len(raw_sentence) > 0:
	        sentences.append(document_to_wordlist( raw_sentence, \
	          remove_stopwords ))
	return sentences,raw_sentences

def load_word2vec(dir):
	'''
		reads word vector files and returns a dictionary of 
		word and assosiated vector as key value pair
	'''
	word2vec = {}
	for path in os.listdir(dir):
		iword2vec = {}
		#load the word2vec features.
		with open(os.path.join(dir,path), 'r') as fin:
			if path == 'vectors0.txt':
				next(fin) #skip information on first line
			for line in fin:
				items = line.replace('\r','').replace('\n','').split(' ')
				if len(items) < 10: continue
				word = items[0]
				vect = np.array([float(i) for i in items[1:] if len(i) > 1])
				iword2vec[word] = vect
		
		word2vec.update(iword2vec)
		
	return word2vec

@app.route('/')
def home():
	''' render a beautiful templete that takes N number
		of documents and Algo option.
	'''
	return render_template('home.html') 


@app.route('/visual',methods=['POST'])
def visual():
	''' get data depending on algo create a similarity matrix
		feed it to a templete for visulization
	'''
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	ALGORITHM = request.form['algorithm']
	LEVEL = request.form['level']
	DOC_COUNT = int(request.form['num-of-docs'])

	DOCUMENTS = []
	for i in range(DOC_COUNT):
		DOCUMENTS.append(request.form['document'+str(i+1)])

	raw_sentences = []

	if LEVEL=="sentence":  
		for each in DOCUMENTS:	# raw sentences will be each document splited into sentences
			raw_sentences+=tokenizer.tokenize(each.decode('utf8').strip())
	else:
		raw_sentences = DOCUMENTS # raw sentence will be the whole do itself.
	matrix = []
	if ALGORITHM=="TF-IDF":
	# Need to write functions for each. Wrote for TF-IDF.
		tfidf = TfidfVectorizer().fit_transform(raw_sentences)
		matrix = (tfidf * tfidf.T).A

	# For each algo the Idea is to form a martix of similarities.
	#---------
		#Algo 2:Latent Semantic Indexing
	
	if ALGORITHM=="LSI": 
		#added by sneha git:coder477 .
		texts = [] 
		matrix = np.zeros(shape=(len(raw_sentences), len(raw_sentences)))
		for each in raw_sentences:
			texts.append(document_to_wordlist(each))
		
		dictionary = corpora.Dictionary(texts)
		corpus = [dictionary.doc2bow(text) for text in texts]
		lsii = models.LsiModel(corpus)
		
		matrix = np.zeros(shape=(len(raw_sentences), len(raw_sentences)))

		for i in range(len(raw_sentences)):
		    vec = corpus[i]
		    doc = raw_sentences[i]
		    
		    vec_bow = dictionary.doc2bow(doc.lower().split())
		    vec_lsi = lsii[vec_bow]  # convert the query to LSI space

		    index = similarities.MatrixSimilarity(lsii[corpus])
		    sims = index[vec_lsi]  # perform a similarity query against the corpus
		    cosine = list(enumerate(sims))
		    for j in range(len(raw_sentences)):
		        matrix[i][j] = cosine[j][1]


	#---------
		#Algo 3
	if ALGORITHM == "WORDNET":
		print("here---------------------------")
		matrix = []
		for each in range(len(raw_sentences)):
			li= []
			for each1 in range(len(raw_sentences)):
				li.append(0)
			matrix.append(li)
		for i in range(0,len(raw_sentences)):
			for j in range(0,len(raw_sentences)):
				input1=raw_sentences[i].encode('ascii','ignore')
				input2=raw_sentences[j].encode('ascii','ignore')

				input1_nps=nps(input1)
				input2_nps=nps(input2)
				common_nps=common1(input1_nps,input2_nps)
				lsv_input1=lsv(common_nps,input1_nps)
				lsv_input2=lsv(common_nps,input2_nps)
				matrix[i][j] =cosine_similarity(lsv_input1,lsv_input2)
	#---------
		#Algo 4
		#Got pretrained vectors from GIT. 
                #added by sneha git:coder477 .
	if ALGORITHM=="WORD2VEC":
		word_vector = load_word2vec('static\\vectors')
		matrix = []
		for each in range(len(raw_sentences)):
			li=[]
			for each1 in range(len(raw_sentences)):
				li.append(0)
			matrix.append(li)
		for i in range(0,len(raw_sentences)):
			for j in range(0,len(raw_sentences)):
				sen1 = raw_sentences[i]
				sen2 = raw_sentences[j]
				sen1_words = document_to_wordlist(sen1)
				sen2_words = document_to_wordlist(sen2)
				sen1_vectors = []
				for each in sen1_words:
					if each in word_vector:
						sen1_vectors.append(word_vector[each])
				sen1_vector = np.array(sen1_vectors).sum(axis=0)
				sen2_vectors = []
				for each in sen2_words:
					if each in word_vector:
						sen2_vectors.append(word_vector[each])
				sen2_vector = np.array(sen2_vectors).sum(axis=0)
				matrix[i][j] = cosine_similarity(sen1_vector, sen2_vector)[0][0]

	#---------
	#Forming nodes and links for graph.
	#code might as well be same for all algos.
	#Refine note : Think of creating private funcs and moving code.
	force = {}
	force["nodes"] = []
	force["links"] = [] 
	for each in raw_sentences:
	    temp={}
	    temp["name"] = each
	    temp["length"] = len(document_to_wordlist(each))
	    force["nodes"].append(temp)
	for ((i,_),(j,_)) in itertools.combinations(enumerate(raw_sentences), 2):
	    temp = {}
	    temp["source"] = i
	    temp["target"] = j
	    temp["value"] = matrix[i][j]
	    force["links"].append(temp)
	graph = json.dumps(force)
	wordlist = []
	for each in raw_sentences:
		wordlist+=document_to_wordlist(each)
	c = Counter(wordlist)
	wordcloud = []
	for each in c:
	    temp = {}
	    temp["text"] = each
	    temp["size"] = c[each]*20
	    wordcloud.append(temp)
	wordcloud = json.dumps(wordcloud)
	return render_template('visual.html', graph=graph, sentences=raw_sentences, wordcloud=wordcloud)


if __name__ == '__main__':

	app.debug = True
	
	app.run()
