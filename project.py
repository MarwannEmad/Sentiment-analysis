import pickle
import streamlit as st
import nltk # Text libarary
# nltk.download('stopwords')
import string # Removing special characters {#, @, ...}
import re as re # Regex Package
from nltk.corpus import stopwords # Stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer # Stemmer & Lemmatizer
from gensim.utils import simple_preprocess  # Text ==> List of Tokens

model_path = 'F:/Career Projects/rf_model.pk'
vectorizer_path = 'F:/Career Projects/tfidf_vectorizer.pk'
model = pickle.load(open(model_path,'rb'))
vectorizer = pickle.load(open(vectorizer_path,'rb'))

def Cleaning(strr):
    strr = re.sub("\\<.*?\\>", "", strr)  # Removing tags html from string
    strr = re.sub(r"http\S+", '', strr)  # Removing html links from string
    strr = re.sub(r"www\S+" , '', strr)  # Removing www links from string
    strr = re.sub("&[a-z0-9]+|&#[0-9]{1,6}|&#x[0-9a-f]{1,6}", '',strr) # Removing unrecognized words 
    strr = re.sub("\\s*\\b(?=\\w*(\\w)\\1{2,})\\w*\\b",' ',strr) # Removing repeated character   
    strr = re.sub("[^A-Za-z']+", " ", strr)   # Removing any brackets or special character or numbers 
    strr = re.sub(r'(Mr|Ms|Mrs)\.?\s[A-Z]\w*', " ", strr) # Removing names from strings 
    strr = " ".join(strr.split()) # adjusting the spaces in string
    return strr

def Updated_text(strr):
    strr = re.sub(r"won't", "will not", strr)
    strr = re.sub(r"can\'t", "can not", strr)
    strr = re.sub(r"n\'t", " not", strr)
    strr = re.sub(r"\'re", " are", strr)
    strr = re.sub(r"\'s", " is", strr)
    strr = re.sub(r"\'d", " would", strr)
    strr = re.sub(r"\'ll", " will", strr)
    strr = re.sub(r"\'t", " not", strr)
    strr = re.sub(r"\'ve", " have", strr)
    strr = re.sub(r"\'m", " am", strr)
    return strr


def stopWords(text):
    x = ' '
    lst=list()
    for word in text.split():
        if word not in stop_words:
            lst.append(word)
    return x.join(lst)
    

def lemmatization(text):
    x = ' '
    ls = list()
    for word in text.split():
        ls.append(lemmatizer.lemmatize(word))
    return x.join(ls)


def stemming(text):
    x = ' '
    ls = list()
    for word in text.split():
        ls.append(stemmer.stem(word))
    return x.join(ls)

stemmer = SnowballStemmer("english")
lemmatizer= WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.remove('not')
def prediction(review, model, vectorizer):
    # Clean Review
    review_c = Cleaning(review)
    review_c = Updated_text(review_c)
    review_c = stopWords(review_c)
    review_c = lemmatization(review_c)
    review_c = stemming(review_c)
    # Embed review using tf-idf vectorizer
    embedding = vectorizer.transform([review_c])
    # Predict using your model
    prediction =model.predict(embedding)
    # Return the Sentiment Prediction
    return "Positive" if prediction == 1 else "Negative"

def main() :
    
    primaryColor=""
    base="dark"
    from PIL import Image
    img = Image.open("Amazone_image.png")
    st.image(img, use_column_width=True)
    primaryColor="red"
    st.title('Amazon Food Review')
    st.subheader("Analyze Your Text")
    text = st.text_area('Enter your Review',height=35)
    if st.button('Analyse'):
        pred= prediction(text,model,vectorizer)
        if pred =='Positive':
            return st.success(pred)
        else :
            return st.error(pred)


if __name__=='__main__': 
    main()
