import socket, os, shutil, time
from datetime import datetime
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"
s = socket.socket()
port = 12345
s.bind(('', port))
s.listen(5)
c, addr = s.accept()
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('E:/files/JASON backup 2021-2-23/JASON/NLP response/chatbot_model.h5')
import json
import random
intents = json.loads(open('E:/files/JASON backup 2021-2-23/JASON/NLP response/intents.json').read())
words = pickle.load(open('E:/files/JASON backup 2021-2-23/JASON/NLP response/words.pkl','rb'))
classes = pickle.load(open('E:/files/JASON backup 2021-2-23/JASON/NLP response/classes.pkl','rb'))

import socket
import tqdm
import os



def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.87
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    try:
        res = getResponse(ints, intents)
        return res
    except:
        na="null"
        c.send(na.encode())

    


def sr(information):
    server_log=open("E:/files/server.log", "a+")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    string="["+dt_string+"]"+" "+information+"\n"
    server_log.write(string)
    server_log.close()
    print(string)
    c.send(string.encode())
    
print ("Socket Up and running with a connection from",addr)
while True:
    try:
        rcvdData = c.recv(1024).decode()
        print ("S:",rcvdData)
        if rcvdData!="":
            command=rcvdData
            if command=="inactive mode":
                sr("entering inactive mode")
                a=40
            elif command=="active mode":
                sr("active mode")
                a=1
            elif command=="sendfile":
                a="please send file"
                c.send(a.encode())
                client_socket=c
                address=addr
                received = client_socket.recv(BUFFER_SIZE).decode()
                filename, filesize = received.split(SEPARATOR)
                # remove absolute path if there is
                filename = os.path.basename(filename)
                # convert to integer
                filesize = int(filesize)
                # start receiving the file from the socket
                # and writing to the file stream
                progress = tqdm.tqdm(range(filesize), f"Receiving {filename}", unit="B", unit_scale=True, unit_divisor=1024)
                with open(filename, "wb") as f:
                    while True:
                        # read 1024 bytes from the socket (receive)
                        bytes_read = client_socket.recv(BUFFER_SIZE)
                        if not bytes_read:    
                            # nothing is received
                            # file transmitting is done
                            break
                        # write to the file the bytes we just received
                        f.write(bytes_read)
                        # update the progress bar
                        progress.update(len(bytes_read))
                f.close()
                # close the client socket
                client_socket.close()
                # close the server socket
                s.close()
                path="D:/OneDrive/桌面/programs/"
                dirs=os.path.basename(filename)
                if dirs=="index.html":
                    os.remove("E:/index.html")
                    shutil.move(path+dirs,"E:/")
                    sr("received and moved"+" {"+dirs+"}")
                elif dirs not in os.listdir("E:/files/uploaded/"):
                    shutil.move(path+dirs,"E:/files/uploaded/")
                    sr("received and moved"+" {"+dirs+"}")
                else:
                    os.remove("E:/files/uploaded/"+dirs)
                    shutil.move(path+dirs,"E:/files/uploaded/")
                    sr("received and moved"+" {"+dirs+"}")
            elif command=="clear uploaded":
                sr("uploaded folder cleared")
                shutil.rmtree("E:/files/uploaded/")
                os.mkdir("E:/files/uploaded/")
            elif command!=None and command!="null" and command!='':
                res=chatbot_response(command)
                if res!=None:
                    print("N: "+str(res))
                    c.send(res.encode())
                res=''
            else:
                sr("Nah")
        else:
            time.sleep(2)
            print("disconnected")
            s = socket.socket()
            port = 12345
            s.bind(('', port))
            s.listen(5)
            c, addr = s.accept()
            

    except:
        print("disconnected")
        time.sleep(3)
        s = socket.socket()
        port = 12345
        s.bind(('', port))
        s.listen(5)
        c, addr = s.accept()
c.close()
