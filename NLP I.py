import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
whole=[]
words=[]
define=[]
removed_feature=['PR','CC','RB','MD','JJ']
from nltk.tokenize import RegexpTokenizer
question=0
def convert_to_string(array):
    string = ' '.join([str(elem) for elem in array])
    return string
def get_rid_punctuation(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    sentence = tokenizer.tokenize(sentence)
    return sentence
def process_content(text):
    global whole,words,define,question
    if text[-1]=='?':
        question=1
    tokenized = get_rid_punctuation(text)
    try:
        for i in tokenized[:len(tokenized)]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            whole.append(tagged)
    except Exception as e:
        print(str(e))
    lengthofarray=int(len(whole))
    for i in range(lengthofarray):
        if (whole[i][0][1] not in removed_feature):
            words.append(whole[i][0][0])
    words.pop(0)
    print(whole)
    if question==1:
        words.append("?")
    return words
if __name__=="__main__":  
    text="Is it really interesting to voluntarily waste your valuable time on meaningless and uneducational video games?"
    process_content(text)
    print(convert_to_string(words))
