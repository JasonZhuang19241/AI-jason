import win32com.client as wincl
def say(ax):
    speak = wincl.Dispatch("SAPI.SpVoice")
    speak.Speak(ax)
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
