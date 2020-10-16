import os
import shutil
a=0
for file in os.listdir('known_faces'):
    a+=1
print(a)
names=os.listdir('known_faces')
for i in range(a):
    dirName=names[i]
    print(dirName)
    original=dirName
    dirName=dirName[:-6]
    os.mkdir('known_faces/'+dirName)
    froam='known_faces/'+original
    to='known_faces/'+dirName
    shutil.move(froam, to)
    print("Directory " , dirName ,  " Created ")
