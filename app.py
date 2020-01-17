# Importing Libraries
from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	
	dataset = pd.read_csv('Feedback.tsv', delimiter = '\t') 

	import re 
	
	import nltk 

	nltk.download('stopwords') 

	from nltk.corpus import stopwords 
	
	from nltk.stem.porter import PorterStemmer 

	corpus = [] 
	
	for i in range(0, len(dataset)): 
		
	    review = re.sub('[^a-zA-Z]',' ', dataset['Feedback'][i]) 
	    
	    review = review.lower() 

	    review = review.split() 
	    
	    ps = PorterStemmer() 
   		
	    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
	    
	    review = ' '.join(review) 
	    
	    corpus.append(review)	

	
	from sklearn.feature_extraction.text import CountVectorizer 

	cv = CountVectorizer(max_features = 1500) 

	X = cv.fit_transform(corpus).toarray() 

	
	y = dataset.iloc[:, 1].values 
	
	from sklearn.model_selection import train_test_split 
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
 	
	from sklearn.ensemble import RandomForestClassifier 
	
	model = RandomForestClassifier(n_estimators = 501, criterion = 'entropy') 
	model.fit(X_train, y_train) 

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		my_prediction = model.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)