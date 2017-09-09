import os
from Tkinter import *
from tkFileDialog import askopenfilename

root = Tk()
root.withdraw()
root.update()
imagename = askopenfilename()
root.destroy()


imagename = os.path.split(imagename)[-1]
