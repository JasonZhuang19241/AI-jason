timeofa=0
from difflib import SequenceMatcher
import random
import os
import time
import xlrd
from xlutils.copy import copy    
from xlrd import open_workbook
import cv2
import numpy as np
import sys

names=['search','see','speedtest','wechat','Record','Play','convert','Ul','idm','Time','browser','video','everything','google','youtube','scie','ebook','mmass','periodicvideos','periodictable','textbook','weather','joke','timer','greeting','calculator','music','vscode','shutdown','concentrate']
try:
    import speech_recognition
except:
    os.system("pip install SpeechRecognition")

wb1 = xlrd.open_workbook('social.xls')
sheetx = wb1.sheet_by_name('Sheet 1')
rown=int(sheetx.nrows)
commandnames=sheetx.row_values(0)
commandnumsa=int(sheetx.ncols)
socials=sheetx.col_values(0)
answers=sheetx.col_values(1)

wb2 = xlrd.open_workbook('commands.xls')
sheet1 = wb2.sheet_by_name('Sheet 1')
rown=int(sheet1.nrows)
commandnames=sheet1.row_values(0)
commandnumsa=int(sheet1.ncols)

for i in range(commandnumsa):
    array=sheet1.col_values(i)
    exec(commandnames[i]+"="+str(array))
a=1

def cmp(string, command, c):
    realness=False
    wb2 = xlrd.open_workbook('commands.xls')
    sheet1 = wb2.sheet_by_name('Sheet 1')
    for i in range(len(command)):
        a=command[i]
        s = SequenceMatcher(None, string, a)
        if s.ratio()>0.7:
            if s.ratio()<1.0:
                ai=names.index(c)
                array=sheet1.col_values(ai)
                ax=0
                for j in range(len(array)):
                    if array[j]=='':
                        ax+=1
                for z in range(ax):
                    array.remove('')
                length=len(array)
                book_ro = open_workbook("commands.xls")
                book = copy(book_ro)  # creates a writeable copy
                sheet12 = book.get_sheet(0)  # get a first sheet
                sheet12.write(length,ai,string)
                book.save("commands.xls")
            realness=True
            break
    if string=='' or string==' ':
        realness=False
    return realness
import time
mytime = time.localtime()
if mytime.tm_hour < 12:
    daytime='morning'
elif mytime.tm_hour<18:
    daytime='afternoon'
else:
    daytime='evening'
def say(ax):
    speak = wincl.Dispatch("SAPI.SpVoice")
    speak.Speak(ax)
if a==1:
    print(
        '''
                           ================       ===============       ===============        ================        ===             == 
                           ================       ===============       ===============        ================        == =            ==
                                 ===              ==           ==       ==                     ==            ==        ==  =           ==
                                 ===              ==           ==       ==                     ==            ==        ==   =          ==
                                 ===              ==           ==       ==                     ==            ==        ==    =         ==
                                 ===              ==           ==       ==                     ==            ==        ==     =        ==
                                 ===              == ========= ==       ===============        ==            ==        ==      =       ==
                                 ===              == ========= ==       ===============        ==            ==        ==       =      ==
                                 ===              ==           ==                    ==        ==            ==        ==        =     ==
                                 ===              ==           ==                    ==        ==            ==        ==         =    ==
                           ===   ===              ==           ==                    ==        ==            ==        ==          =   ==
                           ===   ===              ==           ==                    ==        ==            ==        ==           =  ==
                           =========              ==           ==       ===============        ================        ==            = ==
                           =========              ==           ==       ===============        ================        ==             ===
    '''
        )
    import os
    print("welcome!Jason")
    import win32com.client as wincl
    import win32api
    speak = wincl.Dispatch("SAPI.SpVoice")
    speak.Speak("Good "+daytime)

    def mass():
        import re
        import win32com.client as wincl
        speak = wincl.Dispatch("SAPI.SpVoice")
        speak.Speak("mass calculating system uploaded")
        atomic_mass = {
            "H": 1.0079, "He": 4.0026, "Li": 6.941, "Be": 9.0122,
            "B": 10.811, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
            "Ne": 20.180, "Na": 22.990, "Mg": 24.305, "Al": 26.982,
            "Si": 28.086, "P": 30.974, "S": 32.065, "Cl": 35.453,
            "Ar": 39.948, "K": 39.098, "Ca": 40.078, "Sc": 44.956,
            "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
            "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546,
            "Zn": 65.39, "Ga": 69.723, "Ge": 72.61, "As": 74.922,
            "Se":78.96, "Br": 79.904, "Kr": 83.80, "Rb": 85.468, "Sr": 87.62,
            "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.94,
            "Tc": 97.61, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42,
            "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71,
            "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29,
            "Cs": 132.91, "Ba": 137.33, "La": 138.91, "Ce": 140.12,
            "Pr": 140.91, "Nd": 144.24, "Pm": 145.0, "Sm": 150.36, "Eu": 151.96,
            "Gd": 157.25, "Tb": 158.93, "Dy": 162.50, "Ho": 164.93, "Er": 167.26,
            "Tm": 168.93, "Yb": 173.04, "Lu": 174.97, "Hf": 178.49, "Ta": 180.95,
            "W": 183.84, "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 196.08,
            "Au": 196.08, "Hg": 200.59, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98,
            "Po": 209.0, "At": 210.0, "Rn": 222.0, "Fr": 223.0, "Ra": 226.0,
            "Ac": 227.0, "Th": 232.04, "Pa": 231.04, "U": 238.03, "Np": 237.0,
            "Pu": 244.0, "Am": 243.0, "Cm": 247.0, "Bk": 247.0, "Cf": 251.0, "Es": 252.0,
            "Fm": 257.0, "Md": 258.0, "No": 259.0, "Lr": 262.0, "Rf": 261.0, "Db": 262.0,
            "Sg": 266.0, "Bh": 264.0, "Hs": 269.0, "Mt": 268.0
        }

        def find_closing_paren(tokens):
            count = 0
            for index, tok in enumerate(tokens):
                if tok == ')':
                    count -= 1
                    if count == 0:
                        return index
                elif tok == '(':
                    count += 1
                    
            raise ValueError('unmatched parentheses')

        def parse(tokens, stack):
            if len(tokens) == 0:
                return sum(stack)
            tok = tokens[0]
            if tok == '(':
                end = find_closing_paren(tokens)
                stack.append(parse(tokens[1:end], []))
                return parse(tokens[end + 1:], stack)
            elif tok.isdigit():
                stack[-1] *= int(tok)
            else:
                stack.append(atomic_mass[tok])
            return parse(tokens[1:], stack)
        formula=0
        while True and formula!='end':
            formula = input('Enter molecular formula: ')
            tokens = re.findall(r'[A-Z][a-z]*|\d+|\(|\)', formula)
            print('The molecular mass of {} is {:.3f}\n'.format(formula, parse(tokens, [])))
        import win32com.client as wincl
        speak = wincl.Dispatch("SAPI.SpVoice")
        speak.Speak("molar mass calculator aborted")
    def record():
        import pyaudio
        import wave

        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 2
        fs = 44100  # Record at 44100 samples per second
        seconds = 3
        filename = "output.wav"

        p = pyaudio.PyAudio()  # Create an interface to PortAudio

        print(':r')

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        frames = []  # Initialize array to store frames

        # Store data in chunks for 3 seconds
        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        # Stop and close the stream 
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()

        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()
    command=0
    oral=1
    failed_time=0
    while command!="bye" and command!="goodbye" and command!="good bye" and timeofa<30:
        timeofa+=1
        if command=='oral' or oral==1:
            oral=1
            failed_time=0
        if (command!='oral' and oral==0) or failed_time>=6:
            oral=0
        if oral==1:
            failed_time=0
            record()
            try:
                import speech_recognition as sr
                filename = "output.wav"
                r = sr.Recognizer()
                with sr.AudioFile(filename) as source:
                    audio_data = r.record(source)
                    text = r.recognize_google(audio_data)
                    print(text)
                failed_time=0
            except:
                abscaf=2
                text=''
                failed_time+=1
            command=text
        elif oral==0:
            command=input(":")
        if command=='stop listening':
            oral=0
            say("I will await further commands")
        
        
            try:
                record()
                import speech_recognition as sr
                filename = "output.wav"
                r = sr.Recognizer()
                with sr.AudioFile(filename) as source:
                    audio_data = r.record(source)
                    text = r.recognize_google(audio_data)
                    print(text)
            except:
                text=''
        if command=='play game':
            os.startfile("russian.py")
        if cmp(command,search,'search')==True:
            try:
                print("importing wikipedia...")
                import wikipedia
                print("wikipedia ready")
                import win32com.client as wincl
                speak = wincl.Dispatch("SAPI.SpVoice")
                speak.Speak("I am ready for searching")
                context=input("search:")
                a=context.capitalize()
                print("processing...")
                print(wikipedia.summary(a))
                yorno=input("do you want full output? 'yes' or 'no'... ")
                while yorno!='yes' and yorno!='no' and yorno!='pdf':
                    yorno=input("do you want full output? 'yes' or 'no'... ")
                if yorno=='pdf':
                    import webbrowser
                    pdf="https://en.wikipedia.org/api/rest_v1/page/pdf/"+a
                    webbrowser.open(pdf)
                if yorno=='yes':
                    info=wikipedia.page(a).content
                    from docx import Document
                    document = Document()
                    document.add_paragraph(info)
                    document.save('wikipedia.docx')
                    os.startfile('wikipedia.docx')
                    say("wikipedia page opened")
            except:
                print("sorry, the input is invalid or the internet is not stable")
                say("sorry, an error occurred")

        elif cmp(command,speedtest,'speedtest')==True:
            import win32com.client as wincl
            speak = wincl.Dispatch("SAPI.SpVoice")
            speak.Speak("testing speed")
            os.system('speedtest-cli')
            say("quite a nice speed")
        elif cmp(command,concentrate,'concentrate')==True:
            p=False
            q=False
            lab=1
            lab1=1
            a=0
            CONFIDENCE = 0.5
            SCORE_THRESHOLD = 0.5
            IOU_THRESHOLD = 0.5

            net=cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
            classes=[]
            with open("coco.names","r") as f:
                classes= [line.strip() for line in f.readlines()]
            print(classes)
            layer_names=net.getLayerNames()
            outputlayers=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
            colors=np.random.uniform(0,255,size=(len(classes),3))

            cap=cv2.VideoCapture(0) #0 for 1st webcam
            font = cv2.FONT_HERSHEY_PLAIN
            starting_time= time.time()
            frame_id = 0
            start=time.time()
            while True:
                waste=time.time()
                _,frame= cap.read() # 
                frame_id+=1
                
                height,width,channels = frame.shape
                #detecting objects
                blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False) #reduce 416 to 320    

                    
                net.setInput(blob)
                outs = net.forward(outputlayers)
                #print(outs[1])


                #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
                class_ids=[]
                confidences=[]
                boxes=[]
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.3:
                            #onject detected
                            center_x= int(detection[0]*width)
                            center_y= int(detection[1]*height)
                            w = int(detection[2]*width)
                            h = int(detection[3]*height)

                            #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
        
                            #rectangle co-ordinaters
                            x=int(center_x - w/2)
                            y=int(center_y - h/2)
                            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                            boxes.append([x,y,w,h]) #put all rectangle areas
                            confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                            class_ids.append(class_id) #name of the object tha was detected
                indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
                Labels=[]

                for i in range(len(boxes)):
                    if i in indexes:
                        x,y,w,h = boxes[i]
                        label = str(classes[class_ids[i]])
                    
                        Labels.append(label)
                        confidence= confidences[i]
                        color = colors[class_ids[i]]
                        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                        cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)
                        

                elapsed_time = time.time() - starting_time
                fps=frame_id/elapsed_time
                cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)

                if "person" in Labels:
                    p=True
                    if q==True:
                        start=time.time()
                        q=False
                    if lab1==1:
                        print("studying...")
                        print(a)
                        lab1=0
                        lab=1
                else:
                    if p==True:
                        end=time.time()
                        a=a+end-start
                        p=False
                    q=True
                    if lab==1:
                        print('not studying')
                        print(a)
                        lab1=1
                        lab=0
                
                cv2.imshow("Image",frame)
                key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
                
                if key == 27: #esc key stops the process
                    if p==True:
                        end=time.time()
                        a=a+end-start
                    break;
                
            cap.release()    
            cv2.destroyAllWindows()
            file = open('study.txt', 'r')
            reading=file.read()
            file.close()
            file = open('study.txt', 'w')
            ax=int(reading)+int(a)
            h=ax//3600
            m=(ax-3600*h)//60
            s=ax-3600*h-60*m
            print(str(h)+":"+str(m)+":"+str(s))
            file.write(str(ax))
            file.close()
            say("finished")
            
        elif cmp(command,wechat,'wechat')==True:
            print('opening wechat...')
            os.startfile("C:/Program Files (x86)/Tencent/WeChat/WeChat.exe")
            print('wechat opened')
            import win32com.client as wincl
            speak = wincl.Dispatch("SAPI.SpVoice")
            speak.Speak("wechat opened")
        elif cmp(command,see,'see')==True:
            os.startfile('object_detection.py')
        elif cmp(command,Record,'Record')==True:
            say('recording started')
            import pyaudio
            import wave

            chunk = 1024
            sample_format = pyaudio.paInt16
            channels = 2
            fs = 44100
            seconds = 5
            filename = "recording.wav"

            p = pyaudio.PyAudio()

            print(':')

            stream = p.open(format=sample_format,
                            channels=channels,
                            rate=fs,
                            frames_per_buffer=chunk,
                            input=True)

            frames = []
            for i in range(0, int(fs / chunk * seconds)):
                data = stream.read(chunk)
                frames.append(data)
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf = wave.open(filename, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(sample_format))
            wf.setframerate(fs)
            wf.writeframes(b''.join(frames))
            wf.close()
        elif cmp(command,Play,'Play')==True:
            say("this is what you said")
            import sounddevice as sd
            import soundfile as sf
            filename = 'recording.wav'
            data, fs = sf.read(filename, dtype='float32')  
            sd.play(data, fs)
            status = sd.wait()
        elif cmp(command,convert,'convert')==True:
            say("converting")
            import speech_recognition as sr
            filename = "recording.wav"
            r = sr.Recognizer()
            with sr.AudioFile(filename) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data)
                print(text)
            file=open("command.txt","w")
            file.write(str(text))
            file.write("\n")
            file.close()       
            os.startfile('command.txt')
        elif cmp(command,Ul,'Ul')==True:
            url=input("input url: ")
            import webbrowser
            webbrowser.open(url, new=2)
            say("website opened")
        elif cmp(command,idm,'idm')==True:
            os.startfile("C:/Program Files (x86)/Internet Download Manager/IDMan.exe")
            say("download manager opened")
        elif cmp(command,Time,'Time')==True:
            import datetime
            ac=datetime.datetime.now()
            print(ac)
            say(ac)
           
        elif cmp(command,video,'video')==True:
            path = 'D:/download/Video'
            path2='D:/OneDrive/文件：D/videos/In A Nutshell'
            files = []
            files2=[]
            # r=root, d=directories, f = files
            for r, d, f in os.walk(path):
                for file in f:
                    if '.mp4' in file:
                        files.append(os.path.join(r, file))
            for r, d, f in os.walk(path2):
                for file in f:
                    if '.mp4' in file:
                        files2.append(os.path.join(r, file))
            number=1
            for f in files:
                print(str(number)+". "+f)
                number+=1
            number2=1
            for f in files2:
                print(str(number2)+". "+f)
                number2+=1
            nob=input('which one to open?')
            while nob.isnumeric() is False or int(nob)>number+number2 or int(nob)<0:
                nob=input('which one to open?')
            if int(nob)<=number:
                direction=files[int(nob)-1]
            else:
                direction=files2[int(nob)-1-number]
            os.startfile(direction)
        elif cmp(command,everything,'everything')==True:
            print("opening...")
            os.startfile("D:\Program Files\Everything\Everything.exe")
        elif cmp(command,google,'google')==True:
            print("opening google.com")
            import webbrowser
            webbrowser.open('http://www.google.com', new=2)
            print("browser opened")
            say("google opened")
        elif cmp(command,youtube,'youtube')==True:
            print("opening youtube...")
            import webbrowser
            webbrowser.open('https://www.youtube.com/?gl=US&tab=w1', new=2)
            print("youtube opened")
            say("youtube opened")
        elif cmp(command,periodicvideos,'periodicvideos')==True:
            print("opening the requested youtube page...")
            import webbrowser
            webbrowser.open("https://www.youtube.com/watch?v=6rdmpx39PRk&list=PL7A1F4CF36C085DE1", new=2)
            print("opened")
        elif cmp(command,periodictable,'periodictable')==True:
            print("picture opened")
            os.startfile("C:/Users/chenz/OneDrive/文件：D/SCIE/periodic table.jpg")
            say('periodic table opened')
        elif cmp(command,scie,'scie')==True:
            os.startfile("C:/Users/chenz/OneDrive/文件：D/SCIE")
        elif cmp(command,ebook,'ebook')==True:
            os.startfile("C:/Users/chenz/OneDrive/文件：D/SCIE/e-book")
            say("ebooks opened")
        elif cmp(command,mmass,'mmass')==True:
            mass()
        elif cmp(command,textbook,'textbook')==True:
            path = 'C:/Users/chenz/OneDrive/文件：D/SCIE/textbooks'
            files = []
            # r=root, d=directories, f = files
            for r, d, f in os.walk(path):
                for file in f:
                    if '.pdf' in file:
                        files.append(os.path.join(r, file))
            number=1
            for f in files:
                print(str(number)+". "+f)
                number+=1
            nob=input('which one to open?')
            while nob.isnumeric() is False or int(nob)>number or int(nob)<0:
                nob=input('which one to open?')
            direction=files[int(nob)-1]
            os.startfile(direction)
            say("textbooks opened")
        elif cmp(command,weather,'weather')==True:
            import http.client

            conn = http.client.HTTPSConnection("community-open-weather-map.p.rapidapi.com")

            headers = {
                'x-rapidapi-host': "community-open-weather-map.p.rapidapi.com",
                'x-rapidapi-key': "a800af1539mshc6618457b6753c8p1dcbacjsn0ce0aa51f661"
                }

            conn.request("GET", "/weather?callback=test&id=2172797&units=%2522metric%2522%20or%20%2522imperial%2522&mode=html&q=Shenzhen", headers=headers)
            res = conn.getresponse()
            data = res.read()
            print(data.decode("utf-8"))
            say("this is the weather report")
        elif cmp(command,joke,'joke')==True:
            wb2 = xlrd.open_workbook('jokes.xls')
            sheet1 = wb2.sheet_by_name('Sheet 1')
            rown=int(sheet1.nrows)
            commandnames=sheet1.row_values(0)
            commandnumsa=int(sheet1.ncols)
            from random import randrange
            x=randrange(1,rown)
            array=sheet1.col_values(0)
            say(array[x-1])

        elif cmp(command,timer,'timer')==True:
            say("When should i remind you?")
            a=input("When should i remind you?")
            try:
                import Tkinter as tk
            except:
                import tkinter as tk
                
            import time

            class Clock():
                def __init__(self):
                    self.root = tk.Tk()
                    self.label = tk.Label(text="", font=('Helvetica', 48), fg='red')
                    self.label.pack()
                    self.update_clock()
                    self.root.mainloop()

                def update_clock(self):
                    now = time.strftime("%H:%M:%S")
                    if now>=a:
                        say("time is up")
                    self.label.configure(text=now)
                    self.root.after(1000, self.update_clock)

            app=Clock()
        elif cmp(command,greeting,'greeting')==True:
            say("Hello to you too")
        elif cmp(command,shutdown,'shutdown')==True:
            a=print("what is the secrete code")
            say("what is the code")
            try:
                record()
                import speech_recognition as sr
                filename = "output.wav"
                r = sr.Recognizer()
                with sr.AudioFile(filename) as source:
                    audio_data = r.record(source)
                    text = r.recognize_google(audio_data)
                    print(text)
            except:
                text=''
            if text!="my english teacher is a big fat pig":
                os.system("shutdown -p")
            else:
                print("access denied")
                say("access denied")
        elif cmp(command,music,'music')==True:
            path = 'C:/Users/chenz/Music/Jarvis'
            files = []
            for r, d, f in os.walk(path):
                for file in f:
                    if '.mp3' in file:
                        files.append(os.path.join(r, file))
            number=1
            say("i will choose one for you")
            for f in files:
                print(str(number)+". "+f)
                number+=1
            num1 = random.randint(0, number)
            pather=files[num1]
            os.startfile(pather)
        elif cmp(command,vscode,'vscode')==True:
            os.startfile("C:/Program Files (x86)/Microsoft Visual Studio/2019/Professional/Common7/IDE/devenv.exe")
            say("VS code opened")
        elif command in socials:
            indexr=socials.index(command)
            say(answers[indexr])
        elif cmp(command,calculator,'calculator')==True:
            import tkinter
            tk = tkinter.Tk()
            tk.geometry('300x210+500+200')
            tk.resizable(False, False)
            tk.title('计算器')
            contentVar = tkinter.StringVar(tk, '')
            contentEntry = tkinter.Entry(tk, textvariable=contentVar)
            contentEntry['state'] = 'readonly'
            contentEntry.place(x=20, y=10, width=260, height=30)
            bvalue = ['C', '+', '-', '//', '2', '0', '1', '√', '3', '4', '5', '*', '6', '7', '8', '.', '9', '/', '**', '=']
            index = 0
            for row in range(5):
                for col in range(4):
                    d = bvalue[index]
                    index += 1
                    btnDigit = tkinter.Button(tk, text=d, command=lambda x=d: onclick(x))
                    btnDigit.place(x=20 + col * 70, y=50 + row * 30, width=50, height=20)
            def onclick(btn):
                operation = ('+', '-', '*', '/', '**', '//')
                content = contentVar.get()
                if content.startswith('.'):
                    content = '0' + content 
                if btn in '0123456789':
                    content += btn
                elif btn == '.':
                    lastPart = re.split(r'\+|-|\*|/', content)[-1]
                    if '.' in lastPart:
                        tkinter.messagebox.showerror('错误', '重复出现的小数点')
                        return
                    else:
                        content += btn
                elif btn == 'C':
                    content = ''
                elif btn == '=':
                    try:
                        content = str(eval(content))
                    except:
                        tkinter.messagebox.showerror('错误', '表达式有误')
                        return
                elif btn in operation:
                    if content.endswith(operation):
                        tkinter.messagebox.showerror('错误', '不允许存在连续运算符')
                        return
                    content += btn
                elif btn == '√':
                    n = content.split('.')
                    if all(map(lambda x: x.isdigit(), n)):
                        content = eval(content) ** 0.5
                    else:
                        tkinter.messagebox.showerror('错误', '表达式错误')
                        return
                contentVar.set(content)
            a=input()






