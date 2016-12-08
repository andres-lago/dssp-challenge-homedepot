import re, collections
import HTMLParser
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')

alphabet = 'abcdefghijklmnopqrstuvwxyz'
#remove symbols and numbers and convert to lower case
def words(s):
	h = HTMLParser.HTMLParser()
	s=h.unescape(s)
	s=re.sub('[?!\\.,\(\)#\"\']+',' ',s).strip()
	s=re.sub('[ ]+[0-9]+[/]+[0-9]+[ ]+',' ',s)
	s=re.sub('[ ]+[0-9]+[\-]+[0-9]+[ ]+',' ',s)
	s=re.sub('[ ]+[0-9]+[\.]+[0-9]+[ ]+',' ',s)
	s=re.sub('[ ]+[0-9]+[,]+[0-9]+[ ]+',' ',s)
	s=re.sub('[\-]+',' ',s)
	s=re.sub('[0-9]+',' ',s)
	s=re.sub('([a-z]+)([A-Z]{1,1})([a-z]+)',r'\1 \2\3',s)
	s=re.sub('\s+',' ',s)
	return re.findall('[a-z]+', s.lower())
	


#count occurrences of each word this will be our likelihood
def train(data,field):
	#use given field to count occurrences 
	model=data.flatMap(lambda x:[(y,1) for y in x[field].encode("UTF-8").split() ]).countByKey()
	return model




#create possible versions of a word
def edits1(word):
    s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in s if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]
    inserts    = [a + c + b     for a, b in s for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

#get KNOWN edits of aword
def known_edits2(word,model):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in model.value)

#get KNOWN words
def known(words,model):
    return set(w for w in words if w in model.value)

	
#retrieve correction with maximum likelihood give possible edits
def correct(word,model):
    candidates = known([word],model) or known(edits1(word),model) or    known_edits2(word,model) or [word]
    return max(candidates, key=model.value.get)

