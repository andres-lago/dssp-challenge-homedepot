import argparse
import re #Used for data cleaning
from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SQLContext 
from pyspark.sql import HiveContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover
from pyspark.mllib.linalg import SparseVector
from pyspark.ml.linalg import DenseVector
from pyspark.sql import Row
from functools import partial

# Import learners from PySpark ML package
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

#from numpy import allclose
#from pyspark.ml.linalg import Vectors

# Import text mining tools from Natural Language Tool Kits
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.corpus import stopwords


kStopWList = ["i","a","about","an","are","as","at",\
				 "be","by","com","for","from","how","in","is","it","of","on","or",\
				 "that","the","this","to","was","what","when","where","who","will",\
				 "with","the","www"]

def fixEncoding(x):
	# fix encoding in fields name and value
	id=x['product_uid']
	name=''
	if x['name'] is not None:
		name=x['name'].encode("UTF-8")
	value=""
	if x['value'] is not None:
		value=x['value'].encode("UTF-8")
	retVal='%s %s.'%(name,value)
	#return tuple instead of row 
	return (id,[retVal])
	
def addFeatureLen(row):
	vector=row['tf_idf']
	size=vector.size
	newVector={}
	for i,v in enumerate(vector.indices):
		newVector[v]=vector.values[i]
	newVector[size]=len(vector.indices)
	size+=1
	#we cannot change the input Row so we need to create a new one
	data=row.asDict()
	data['tf_idf']= SparseVector(size,newVector)
	#new Row object with specified NEW fields
	newRow=Row(*data.keys())
	#fill in the values for the fields
	newRow=newRow(*data.values())
	return newRow
	
def cleanData(row,model):
	#we are going to fix search term field
	text=row['search_term'].split()
	for i,v in enumerate(text):
		text[i]=correct(v,model)
	data=row.asDict()
	#create new field for cleaned version
	data['search_term2']=text
	newRow=Row(*data.keys())
	newRow=newRow(*data.values())
	return newRow



### Lower case transformation ###
#in columns: "search_term"  "product_title"
def lowerCaseTransform(row):
	data = row.asDict()

	#
	#search_term
	#

	#Split field into words
	words = word_tokenize(row['search_term'])

	# Lower case words
	trf_words = [word for word in words]
	##Convert from list to string
	trf_words = ' '.join(trf_words).lower()

	# Create new field for cleaned version
	data['search_term_lowerc'] = trf_words

	#
	#product_title
	#
	#Split field into words
	words = word_tokenize(row['product_title'])

	# Lower case words
	trf_words = [word for word in words]
	##Convert from list to string
	trf_words = ' '.join(trf_words).lower()

	# Create new field for cleaned version
	data['product_title_lowerc'] = trf_words


	newRow = Row(*data.keys())
	newRow = newRow(*data.values())
	return newRow

### Removal of 'stop words' ###
#in columns: "search_term"  "product_title"
def stopWordRemoval(row):
	data = row.asDict()


	#
	#search_term
	#

	#Split field into words
	words = word_tokenize(row['search_term'])
	#stopWList = set(stopwords.words('english'))

	# Remove stop words
	stopwordsfree_words = [word for word in words if word.lower() not in kStopWList]
	##Convert from list to string
	stopwordsfree_words = ' '.join(stopwordsfree_words)

	# Create new field for cleaned version
	data['search_term_no_stop'] = stopwordsfree_words

	#
	#product_title
	#

	#Split field into words
	words = word_tokenize(row['product_title'])
	#stopWList = set(stopwords.words('english'))

	# Remove stop words
	stopwordsfree_words = [word for word in words if word.lower() not in kStopWList]
	##Convert from list to string
	stopwordsfree_words = ' '.join(stopwordsfree_words)

	# Create new field for cleaned version
	data['product_title_no_stop'] = stopwordsfree_words


	newRow = Row(*data.keys())
	newRow = newRow(*data.values())
	return newRow

### Removes punctuation, changes to lowercase, and strips leading and trailing spaces ###
def removePunctuation(row,model):
	# we are going to fix search term field
	text = row['search_term'].split()
	for i, v in enumerate(text):
		re.sub('[^a-z| |0-9]', '', text[i].strip().lower())
	data['search_term2']=text
	newRow=Row(*data.keys())
	newRow=newRow(*data.values())
	return newRow

#def removePunction(row,model):
#	punctuation=set(string.punctuation)
#	doc = ''.join([w for w in text.lower() if w not in punctuation])
#	return newRow

def newFeatures(row):
	vector=row['tf_idf']
	data=row.asDict()
	data['features']= DenseVector([len(vector.indices),vector.values.min()])
	newRow=Row(*data.keys())
	newRow=newRow(*data.values())
	return newRow

#Add a feature: similarity between product_title and search_term
def addSimilarity(row):
	v1 = row['tf_idf1']
	v2 = row['tf_idf_all']
	#We'll use cosine similarity to compute the difference
	## result is scalar: cosine of the angle between the two vectors
	##     The higher the cosine, the closer the similarity
	cosSim = v1.dot(v2)/(v1.norm(2) * v2.norm(2))

	#print "cosSim: "
	#print cosSim

	data=row.asDict()
	#Add the column 'features'
	#data['features']= DenseVector([len(vector.indices),vector.values])
	data['features']= DenseVector([1, cosSim]) #field can't be float, we convert to vector
	#data['features']= cosSim

	newRow=Row(*data.keys())
	newRow=newRow(*data.values())
	return newRow

def stemString(i_str):
	stemmer = PorterStemmer()
	#stemmedStr = [stemmer.stem(w) for w in i_str] #output will be an array of chars

	words = word_tokenize(i_str)

	stemmedStr = ""
	for w in words:
		stemmedStr = stemmedStr + " " + stemmer.stem(w)

	return stemmedStr

def stemColumns(row):
	#we cannot change the input Row so we need to create a new one
	data=row.asDict()

	#column 'product_title'
	unStemmedStr = row['product_title']
	#stemmedStrList = ""
	#for i,v in enumerate(unStemmedStrList):
	#	stemmedStrList = stemmedStrList + " " + stemString(unStemmedStr)

	data['product_title_stem'] = stemString(unStemmedStr)

	#print "product_title_stem: "
	#print text_stem

	#column 'search_term'
	unStemmedStr = row['search_term']
	data['search_term_stem'] = stemString(unStemmedStr)

	#print "search_term_stem: "
	#print text_stem

	#new Row object with specified NEW fields
	newRow=Row(*data.keys())
	#fill in the values for the fields
	newRow=newRow(*data.values())
	return newRow

#
# MAIN
#

kRegressorName = "RandomForestRegressor"

# parse args
parser = argparse.ArgumentParser(description="Predictor HOME DEPOT")
parser.add_argument("--lightData", action="store_true", default=False, help="[opt] It will use light version of datasets (files in ../data_light)")

args = parser.parse_args()

#arg: lightData
useLightData = False
if(args.lightData):
	useLightData = True

#Spark context
sc = SparkContext(appName="PredictorHomeDepot")

#Set log level of Spark
sc.setLogLevel("WARN")


print "Current regressor: " + kRegressorName


sqlContext = HiveContext(sc)
print "###############"

# get the current directory
import os

fileRepartition = 1

trainFileAbsPath = ""
if(useLightData):
	cwd = os.getcwd()
	trainFileAbsPath = "file://" + cwd + "/../data_light/train.csv"
	fileRepartition = 1
else:
	#HDFS
	trainFileAbsPath = "/dssp/datacamp/train.csv"
	fileRepartition = 100


print "###############"
#READ data
data=sqlContext.read.format("com.databricks.spark.csv").\
	option("header", "true").\
	option("inferSchema", "true").\
	load(trainFileAbsPath).repartition(fileRepartition)
	#load("/dssp/datacamp/train.csv").repartition(100)
print "data loaded - head:"
print data.show(5)
print "################"


attribFileAbsPath = ""
if(useLightData):
	cwd = os.getcwd()
	attribFileAbsPath = "file://" + cwd + "/../data_light/attributes.csv"
	fileRepartition = 1
else:
	#HDFS
	attribFileAbsPath = "/dssp/datacamp/attributes.csv"
	fileRepartition = 100

attributes=sqlContext.read.format("com.databricks.spark.csv").\
	option("header", "true").\
	option("inferSchema", "true").\
	load(attribFileAbsPath).repartition(fileRepartition)
	#load("/dssp/datacamp/attributes.csv").repartition(100)

print "attributes loaded - head:"
print attributes.head()
print "################"

#attributes: 0-N lines per product
#Step 1 : fix encoding and get data as an RDD (id,"<attribute name> <value>")
attRDD=attributes.rdd.map(fixEncoding)
print "new RDD:"
print attRDD.first()
print "################"
#Step 2 : group attributes by product id
attAG=attRDD.reduceByKey(lambda x,y:x+y).map(lambda x:(x[0],' '.join(x[1])))
print "Aggregated by product_id:"
print attAG.first()
print "################"
#Step 3 create new dataframe from aggregated attributes
atrDF=sqlContext.createDataFrame(attAG,["product_uid", "attributes"])
print "New dataframe from aggregated attributes:"
print atrDF.head()
print "################"
#Step 4 join data
fulldata=data.join(atrDF,['product_uid'],'left_outer')
print "Joined Data:"
print fulldata.head()
print "################"

##############
#TF-IDF features:  product_title
##############
#Step 1: split product_title into words
##Add column "words_title"
##A tokenizer converts the input string to lowercase and then splits it by white spaces.
# tokenizer = Tokenizer(inputCol="product_title", outputCol="words_title")
# fulldata = tokenizer.transform(fulldata) #Returns the transformed dataset
# print "Tokenized Title:"
# print fulldata.head()
# print "################"
# #Step 2: compute term frequencies
# ##Add column "tf" for  words_title
# hashingTF = HashingTF(inputCol="words_title", outputCol="tf")
# fulldata = hashingTF.transform(fulldata)
# print "TERM frequencies: (product_title)"
# print fulldata.head()
# print "################"
# #Step 3: compute inverse document frequencies
# ##Add column "tf_idf"  of product_title
# idf = IDF(inputCol="tf", outputCol="tf_idf")
# # While applying HashingTF only needs a single pass to the data, applying IDF needs two passes:
# # First to compute the IDF vector and second to scale the term frequencies by IDF.
# idfModel = idf.fit(fulldata) #IDF Vector
# fulldata = idfModel.transform(fulldata)
#
# print "IDF : (product_title)"
# print fulldata.head()
# print "################"
#
# #Step 4 new features column / rename old
# ##"features" column is a denseVector of tf-idf
# fulldata=sqlContext.createDataFrame(fulldata.rdd.map(newFeatures))
# print "NEW features column :"
# print fulldata.head()
# print "################"
# #Step 5: ALTERNATIVE ->ADD column with number of terms as another feature
# fulldata=sqlContext.createDataFrame(fulldata.rdd.map(addFeatureLen))#add an extra column to tf features
# fulldata=fulldata.withColumnRenamed('tf_idf', 'tf_idf_plus')
# print "ADDED a column and renamed :"
# print fulldata.head()
# print "################"

print "################"
print "#Step 4-4: lower case transformation of: search_term + product_title"
fulldata = sqlContext.createDataFrame(fulldata.rdd.map(lowerCaseTransform))
#Remove original  columns:  search_term
fulldata = fulldata.drop('search_term')
fulldata = fulldata.drop('product_title')

#Rename transformed columns
fulldata = fulldata.withColumnRenamed('search_term_lowerc', 'search_term')
fulldata = fulldata.withColumnRenamed('product_title_lowerc', 'product_title')

print fulldata.head(3)


print "################"
print "#Step 4-5: stop words removal"
fulldata = sqlContext.createDataFrame(fulldata.rdd.map(stopWordRemoval))
#Remove original  columns:  search_term
fulldata = fulldata.drop("search_term")
fulldata = fulldata.drop("product_title")

#Rename transformed columns
fulldata = fulldata.withColumnRenamed('search_term_no_stop', 'search_term')
fulldata = fulldata.withColumnRenamed('product_title_no_stop', 'product_title')

print fulldata.head(3)



print "################"
print "#Step 5: stem product_title & search_term"
#
fulldata = sqlContext.createDataFrame(fulldata.rdd.map(stemColumns))
#Remove original unstemmed columns:  product_title & search_term
fulldata = fulldata.drop("product_title")
fulldata = fulldata.drop("search_term")

#Rename stemmed columns
fulldata = fulldata.withColumnRenamed('product_title_stem', 'product_title')
fulldata = fulldata.withColumnRenamed('search_term_stem', 'search_term')


print fulldata.head(3)


##############
#TF-IDF features:  product_title & search_term
#  (we'll use cosine similarity, comparing tf-idf of search query and product title)
##############
print "#Step 6-1: Compute TF for comparison between search query and product title"

fulldata.registerTempTable("fulldatadf")
concatedField=sqlContext.sql("SELECT id,product_uid,attributes,product_title,search_term,CONCAT(product_title,' ',search_term) as allText FROM fulldatadf")
print concatedField.head()
##Add column: "words of allText: product_title & search_term"
tokenizer = Tokenizer(inputCol="allText", outputCol="words_allText")
concatedField = tokenizer.transform(concatedField)
##Add column: "words of search_term"
tokenizer = Tokenizer(inputCol="search_term", outputCol="words_search_term")
concatedField = tokenizer.transform(concatedField)

##Add column: "TF of words of product_title & search_term"
hashingTF = HashingTF(inputCol="words_allText", outputCol="tf1")
concatedField = hashingTF.transform(concatedField)
##Add column: "TF of words of search_term"
hashingTF = HashingTF(inputCol="words_search_term", outputCol="tf2")
concatedField = hashingTF.transform(concatedField)
print concatedField.head()

print "TERM frequencies:"
print concatedField.head()
print "################"
print "#Step 6-2: compute IDF and create TF-IDF for product_title & search_term "
print '##Add column: "IDF of words of product_title & search_term" '
idf = IDF(inputCol="tf1", outputCol="tf_idf1")
#Compute the inverse document frequency.
# .fit(dataset), where dataset=an RDD of term frequency vectors
idfModel = idf.fit(concatedField)
concatedField = idfModel.transform(concatedField)
concatedField=concatedField.withColumnRenamed('tf_idf1', 'tf_idf_all')
concatedField=concatedField.withColumnRenamed('tf1', 'tf1_all')
concatedField=concatedField.withColumnRenamed('tf2', 'tf1')
concatedField = idfModel.transform(concatedField)
print "TF-IDF: (tf_idf_all: product_title & search_term, tf_idf1:  search_term)"
print concatedField.head(5)
print "################"

#create the df with the information that will be joined to the training dataset
#ptSt_df=sqlContext.createDataFrame(concatedField,["id", "tf_idf_all", "tf_idf1"])
ptSt_idf_df = concatedField.select("id", "tf_idf_all", "tf_idf1")

#Join Data: Add columns "tf_idf_all" & "tf_idf1" to main dataset
## how:str, default inner. One of inner, outer, left_outer, right_outer, leftsemi
fulldata=fulldata.join(ptSt_idf_df,on=['id'],how='left_outer')


#
# Compute similarity using COSINE SIMILARITY
#
print "#Step 6-3: Add feature: compute similarity between product_title AND search_term "
fulldata=sqlContext.createDataFrame(fulldata.rdd.map(addSimilarity))
print "ADDED column 'features': <similarity between product_title AND search_term> "

print fulldata.head(5)

#
#create NEW features & train and evaluate regression model
#
#Step 1: create features
## Only 'label' and 'features' columns are selected
fulldata=fulldata.withColumnRenamed('relevance', 'label').select(['label','features'])

print "######## CREATING THE MODEL ########"
print "RDD with train and test set:  ('label' = relevance)"
print fulldata.head(5)


#Simple evaluation : train and test split
(train,test)=fulldata.rdd.randomSplit([0.8,0.2])

#Initialize regresion model
#lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

rf = RandomForestRegressor(numTrees=300, maxDepth=10, seed=42)

# Fit the model
print "Fitting the model..."

model = rf.fit(sqlContext.createDataFrame(train))

#Apply model to test data
print "Testing the model..."
result = model.transform(sqlContext.createDataFrame(test))

#Compute mean squared error metric
#MSE = result.rdd.map(lambda r: (r['label'] - r['prediction'])**2).mean()
#print("ORIG-Mean Squared Error = " + str(MSE))

# Select (prediction, true label) and compute test error
#evaluator = RegressionEvaluator(
#    labelCol="label", predictionCol="prediction", metricName="rmse")
#rmse = evaluator.evaluate(result)
#print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="mse")
mse = evaluator.evaluate(result)
print("Mean Squared Error (MSE) on test data = %g" % mse)


#rfModel = model.stages[1]
#print(rfModel)  # summary only
