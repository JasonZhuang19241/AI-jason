search=["search","wikipedia","Search","Wikipedia"]
speedtest=["speed","speedtest","Speed","Speedtest","speed test"]
wechat=['open WeChat','wechat','WeChat','chat']
Record=['record','record this','start recording']
Play=['play audio','play']
convert=['convert to text','audio to text','speech recognition','recognition']
Ul=['website']
idm=["idm","IDM",'internet download manager','download manager','download','open IDM', 'open YouTube apps download', 'Download Manager', 'internet download manager', 'open download manager', 'IDM']
Time=['what is the time','time','date','tell me the time','show me the time','what time is it']
browser=['browser','Microsoft Edge']
video=['video','open video','play video']
everything=["search file",'everything']
google=['open Google','Google']
youtube=['open YouTube','YouTube']
scie=['scie','SCIE','school folder','school']
ebook=['ebooks','books','electronic book']
mmass=['molar mass','calculate mass','mass']
periodicvideos=["periodic videos",'periodic video','chem video']
periodictable=['periodic table','the table','table']
textbook=['textbook','electronic book','book']
weather=["weather","what is the weather","how's the weather today",'how is the weather today','Taurus horoscope weather', 'how is the weather','who is the weather', 'how is the weather in Jarvis', 'show me the weather', 'what is the weather like outside', 'what is the current weather', 'open weather', 'Joe Weider','the weather']
joke=['tell me a joke','tell a joke','a joke','joke']
timer=['set a timer','timer','remind me later','set a timer', 'hard timing', 'open up timer', 'timer started', 'start a timer','start timing', 'open the timer']
greeting=['morning', 'hello', 'nice to see you', 'hello Jarvis', 'food afternoon', 'evening', 'good afternoon','hi','good morning']
calculator=['open calculator','calculator']
music=["music",'play a music','choose a song','play music','open music']
vscode=['open vs code', 'open decoder', 'cplusplus', 'icing shoulder', 'Microsoft coder', 'vs code', 'code', 'okay vs', 'vs code open',"Open DNS code","Microsoft colder","code","C++"]
shutdown=["close the computer","shutdown","please close the computer"]
names=['search','see','speedtest','wechat','Record','Play','convert','Ul','idm','Time','browser','video','everything','google','youtube','scie','ebook','mmass','periodicvideos','periodictable','textbook','weather','joke','timer','greeting','calculator','music','vscode','shutdown','names']
see=['object detection','open your eyes','see this','see']
import xlwt
from xlwt import Workbook
wb=Workbook()
sheet=wb.add_sheet('Sheet 1')

for i in range(len(names)):
    a=names[i]
    sheet.write(0,i,a)
    exec("length=len("+a+")")
    for j in range(int(length)):
        exec("b="+a+"["+str(j)+"]")
        sheet.write(j+1,i,b)
wb.save("command.xls")

