from pyspark import SparkContext
import pyspark
from pyspark.conf import SparkConf
from pyspark.sql import SQLContext 
from pyspark.sql import HiveContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.mllib.linalg import SparseVector
from pyspark.ml.linalg import DenseVector
from pyspark.sql import Row
from functools import partial
from pyspark.ml.regression import LinearRegression
from check import *



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
	

	
def cleanData(row,model):
	#we are going to fix search term field
	text=row['search_term'].split()
	for i,v in enumerate(text):
		text[i]=correct(v,model)
	data=row.asDict()
	#create new field for cleaned version
	data['search_term2']= text
	newRow=Row(*data.keys())
	newRow=newRow(*data.values())
	return newRow
	


sc = SparkContext(appName="Example1")

sqlContext = HiveContext(sc)
print "###############"
#READ data
data=sqlContext.read.format("com.databricks.spark.csv").\
	option("header", "true").\
	option("inferSchema", "true").\
	load("/dssp/datacamp/train.csv").repartition(100)
print "data loaded - head:"	
print data.head()
print "################"

attributes=sqlContext.read.format("com.databricks.spark.csv").\
	option("header", "true").\
	option("inferSchema", "true").\
	load("/dssp/datacamp/attributes.csv").repartition(100)
	
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

#EXAMPLE clean data with custom dictionary
#Step 1: build dictionary model
model=sc.broadcast(train(atrDF.rdd,'attributes'))
#apply model to  query terms to correct them based on model dictionary
fulldata=sqlContext.createDataFrame(fulldata.rdd.map(partial(cleanData,model=model)))
print "ADDED a column with new cleaned query terms :"
print fulldata.head()
print "################"