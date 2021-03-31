from tkinter.filedialog import askopenfilenames
import os
import time, datetime
import shutil
Current_Date = datetime.datetime.today().strftime ('%d-%b-%Y')
flag=False
while flag==False:
    filenames = askopenfilenames(title = "Select files you want to sort")
    flag=True
    print(filenames)
    name, extension = os.path.splitext(filenames[0])
    for i in range(len(filenames)):
        name1, extension1 = os.path.splitext(filenames[i])
        if extension1!=extension:
            print("please select the same kind of files")
            filenames = askopenfilenames(title = "Select files you want to sort")
            flag==False
for i in range(len(filenames)):
    direction=os.path.dirname(filenames[i])
    os.rename(filenames[i],direction+"/"+
              extension[1:]+" file"+str(i+1)+" "+Current_Date+extension)
