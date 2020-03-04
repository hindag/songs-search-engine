
import ast
import nltk
from nltk.corpus import stopwords
from collections import Counter 
from random import randint
import tkinter 
from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import *
import nltk
nltk.download('stopwords')
nltk.download('words')
nltk.download('wordnet')
import sys
import os
import operator
import numpy as np
import sys
import os
import operator
import math
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import webbrowser
from nltk.corpus import words
from nltk.corpus import wordnet as wn


def create_tuples( fileName ):
	file_open = open(fileName)
	file_read = file_open.read()
	collection = ast.literal_eval(file_read)
	result = []
	for i in collection:
		record = (i["lyric"], i["sentiment"])
		result.append(record)
	return result


def stop_words_removal(lyrics):
	array = []
	for i in lyrics:
		sentiment = i[1]
		stop = set(stopwords.words('english')) 
		filtered_words = [word for word in i[0].split() if word not in stop]
		combine = (filtered_words , sentiment)
		array.append(combine)
	return array 


def get_words(lyrics):
	all_words = []
	for(words, sentiment) in lyrics:
		all_words.extend(words)
	return all_words


def word_features(List_word):
	List_word = nltk.FreqDist(List_word)
	word_features = List_word.keys()
	return word_features


def extract_features(doc):
	doc_words = set(doc)
	features = {}
	for word in word_features: 
		features['contains(%s)' %word] = (word in doc_words)
	return features 



def stop_word_removal1(lyrics):
    stop = set(stopwords.words('english'))
    filtered_words = [word for word in lyrics.split() if word not in stop]
    return filtered_words
    

def user_sentiment(file):
    file_open = open(file)
    file_read = file_open.read()
    file_split=file_read.split("|")
    result_list=[]
    for lyrics in file_split:
        filtered_lyrics=stop_word_removal1(lyrics)
        output = classifier.classify(extract_features(filtered_lyrics))
        result_list.append(output)
    return result_list

  


lyrics = create_tuples(".../training_original.txt")

filtered_corpus = stop_words_removal(lyrics)

word_features = word_features(get_words(filtered_corpus))

training_set = nltk.classify.apply_features(extract_features,filtered_corpus)

classifier = nltk.NaiveBayesClassifier.train(training_set)




lyrics_test = create_tuples(".../testing_original.txt")

test_corpus = stop_words_removal(lyrics_test)

test_set = nltk.classify.apply_features(extract_features,test_corpus)



def tokenise_normalise(raw_string):



	tokenizer = RegexpTokenizer(r'\w+')
	tokenised_text = tokenizer.tokenize(str(raw_string))

	token_normalise = []
	for w in tokenised_text:
				
		w = w.lower()
		
		token_normalise.append(w)
	
	return token_normalise
	
def stem(token_normalised_text):



	processed_text = []
	stemmer = PorterStemmer()

	for w in token_normalised_text:
				
		root = stemmer.stem(w)
		root = str(root)
		
		processed_text.append(root)
	
	return processed_text
	
	
def create_tf_dict_doc(processed_text, f, stats_dict):
		

		
	for root in processed_text:
	
		if root not in stats_dict:
			
			stats_dict[root] = {}
			stats_dict[root][f] = 1
		
		if f not in stats_dict[root]:
			stats_dict[root][f] = 1
			
		stats_dict[root][f] += 1
		
	return stats_dict
		
def create_tf_dict_query(processed_query):


	query_dict = {}
	for root in processed_query:
		
		if root not in query_dict:
			
			query_dict[root] = 1
			
		query_dict[root] += 1
		
	return query_dict
		
		
def find_tfidf_doc(stats_dict, Number_of_docs):


	
	idf_dict = {}
	for word in stats_dict:
		idf_dict[word] = len(stats_dict[word].keys())
		
	for word in stats_dict:
		for doc in stats_dict[word]:
			stats_dict[word][doc] = (1 + math.log(stats_dict[word][doc]))*math.log(Number_of_docs/idf_dict[word])

	return stats_dict, idf_dict
	
def find_tfidf_query(query_dict, idf_dict, Number_of_docs):



	for word in query_dict:
		query_dict[word] = (1 + math.log(query_dict[word]))*math.log(Number_of_docs/idf_dict[word])

	return query_dict


def spell_check(string):

	checker = words.words()
	error_dict = {}
	
	for w in checker:
		
		if(w==False):
			if w not in error_dict:
				error_dict.append(w) 
		
	return error_dict


def synonym_list(processed_query):
.
	final_query_words = []
	for w in processed_query:
	
		syns = wn.synsets(w)
		names=[s.name().split('.')[0] for s in syns]

		if len(names)>=2:names=names[:2]
		names.append(w)
		for n in names:
			final_query_words.append(n)
			
	return final_query_words


def sentiment(output):
    if str(output) == "P":
        return "Positive"
    if str(output) == "N":
        return "Negative"
    if str(output) == "A":
        return ("Sentiment not defined")


def searchQuery(query,text_files,stats_dict,Number_of_docs,doc_list,idf_dict,list_of_words):
	''' searches the query in stats_dict'''

	processed_query = tokenise_normalise(query)
	output = classifier.classify(extract_features(processed_query))

	

	error_dict = spell_check(processed_query)#checking query for spelling mistakes
	

	for w in error_dict:#suggest word for each spelling mistake.
		response = messagebox.askquestion("Suggestion", "Did you mean "+error_dict[w]+ "?", icon='info')
		if response=="yes":
			#print 'ho gaya'
			for i,q in enumerate(processed_query):
				if q==w:
					processed_query[i]=error_dict[w]
				

		

	final_query = synonym_list(processed_query)

	final_query = stem(final_query)
    
	
	


	

	final_query = [x for x in final_query if x in stats_dict]
	if final_query!=[]:
		
		
		query_dict = create_tf_dict_query(final_query) 

		query_dict = find_tfidf_query(query_dict, idf_dict, Number_of_docs)
		
		vector_array = np.zeros((len(query_dict.keys()), Number_of_docs))

		i=0
		j=0

		query_vector = np.zeros((1, len(query_dict.keys())))

		for w in query_dict:
		
			query_vector[0][i] = query_dict[w]
			for d in doc_list:
				if d not in stats_dict[w]:
					j += 1
				else :
					vector_array[i][j] = stats_dict[w][d]
					j += 1
			i += 1
			j = 0
	
		magnitude = np.linalg.norm(vector_array, axis = 0)
		vector_array = np.divide(vector_array, magnitude)
		vector_array[np.isnan(vector_array)] = 0

		q_magnitude = np.linalg.norm(query_vector, axis = 1)
		query_vector = np.divide(query_vector, q_magnitude)

		dot_product = np.dot(query_vector, vector_array) 
		dot_product = dot_product.tolist()

		final_rank = list(zip(dot_product[0], doc_list))
		final_rank.sort(reverse = True)
		searchf=Tk()
		searchf.wm_title(query +" "+'Systeme de recherche Information')
		blank='           '
		blanklabel=Label(searchf,text=blank*40,font=("ComicSansMS", 10))
		label1 = Label( searchf,text=query+" "+sentiment(output)+'\n\n',font=("ComicSansMS", 20))
		#searchf.configure(bg="gray")
		label1.pack()
		blanklabel.pack()
		def callbacks(event):
			webbrowser.open_new(event.widget.cget("text"))
		for i in final_rank[:20]:
			labl=Label( searchf,text=i[1],font=("ComicSansMS", 12),justify=LEFT, fg="blue", cursor="hand2")
			labl.bind("<Button-1>", callbacks)
			labl.pack()
			
	else :messagebox.showinfo( "!!", "Désolé aucun résultat trouvé", icon='warning')


        
corpus_dir=".../corpus"   

top = Tk()
top.wm_title("Moteur de racherche & Analyse du sentiment")


photo = PhotoImage(file="...\\back3.png")
f = Canvas(top, width=photo.width(), height=photo.height(), bg="yellow")
f.create_image(0, 0, anchor=NW, image=photo)
f.pack()
e1=Entry(top,bd=6,width=80)
e1.insert(END, 'Veuillez Entrer votre recherche')
e1.place(relx=0.5, rely=0.35, anchor=CENTER,height=35, width=520)
output=classifier.classify(extract_features(tokenise_normalise(e1.get())))
if output=='P':
    result = ".../Positif"
else :
    result = ".../Negatif"
text_files = [os.path.join(corpus_dir, f) for f in os.listdir(corpus_dir)]
stats_dict = {}

Number_of_docs = len(text_files)	

doc_list = []

for f in text_files:	
	doc = open(f, 'r')
	lines = [l.strip() for l in doc.readlines()]

	index = 0 

	full_transcript = []

	while True:
		if index >= len(lines):
			break
		line = lines[index]
		full_transcript.append(line)
		index += 1
	#tokenising and stemming text
	processed_text = tokenise_normalise(full_transcript)
	processed_text = stem(processed_text)
	stats_dict = create_tf_dict_doc(processed_text, f, stats_dict)
	doc_list.append(f)


stats_dict, idf_dict = find_tfidf_doc(stats_dict, Number_of_docs)

list_of_words = sorted(stats_dict.keys())
doc_list.sort()




photoSearch = PhotoImage(file = r"...\\search.png") 
photoQuitter = PhotoImage(file = r"...\\quitter.png")  
# Resizing image to fit on button 
photoimage = photoSearch.subsample(5, 5) 
photoimageQ = photoQuitter.subsample(10,10) 
b1=Button(top,text="Recherche",image = photoimage, compound = LEFT,height=30, width=100,command= lambda: searchQuery(e1.get(),text_files,stats_dict,Number_of_docs,doc_list,idf_dict,list_of_words))
b1.place(relx=0.5, rely=0.5, anchor=CENTER)
b2=Button(top,text="Quitter",command=top.destroy,image = photoimageQ, compound = LEFT,height=30, width=100)
b2.place(relx=0.62, rely=0.5, anchor=CENTER)

top.mainloop()
