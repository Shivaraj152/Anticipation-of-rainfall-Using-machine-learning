from tkinter import *
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
import pickle

file3 = "finalized_model_NB.sav"

NB = pickle.load(open(file3, 'rb'))


def showRainfall():
    new = np.array([[float(e1.get()),float(e2.get()),float(e3.get()),float(e4.get()),float(e5.get()),float(e6.get()),float(e7.get()),float(e8.get()),float(e9.get())]])
    Ans = NB.predict(new)
    fin=str(Ans)[1:-1]#IT WILL remove[ ]
    Rainfall.insert(0, fin)
    res=str(fin)
    messagebox.showinfo('total rainfall rate: ',res)
    print('total rainfall rate is:' +res)
    if res=='1':
        messagebox.showinfo('Rainfall type is: ','Drizzle')
        print('Rainfall type is: Drizzle')
    elif res=='2':
        messagebox.showinfo('Rainfall type is: ','virage')
        print('Rainfall type is: virage')
    elif res=='3':
        messagebox.showinfo('Rainfall type is: ','sleet')
        print('Rainfall type is: sleet')
    elif res=='4':
        messagebox.showinfo('Rainfall type is: ','foggy')
        print('Rainfall type is: foggy')
    elif res=='5':
        messagebox.showinfo('Rainfall type is: ','snowfall')
        print('Rainfall type is: snowfall')
    elif res=='6':
        messagebox.showinfo('Rainfall type is: ','Hail')
        print('Rainfall type is: Hail')
        
     

master = Tk()
master.title('Rainfall Prediction')
master.geometry('950x850')
master.config(bg='deep sky blue')

img =Image.open("AdobeStock_366903907.jpeg")
bg = ImageTk.PhotoImage(img)


label = Label(master, image=bg)
label.place(x = 0,y = 0)

w = tk.Label(master, 
		 text=" RAINFALL PREDICTION SYSTEM ",
		 fg = "light sky blue",
		 bg = "cadet blue",
		 font = "Helvetica 20 bold italic")
w.pack()
w.place(x=450, y=0)

label2 = Label(master, text="year", anchor= CENTER, width=25, bg="light sky blue").grid(row=4, column=8)
label3 = Label(master, text="month", anchor= CENTER, width=25, bg="light sky blue").grid(row=5, column=8)
label4 = Label(master, text="day", anchor= CENTER, width=25, bg="light sky blue").grid(row=6, column=8)
label5 = Label(master, text="tempavg", anchor= CENTER, width=25, bg="light sky blue").grid(row=7, column=8)
label6 = Label(master, text="DPavg", anchor= CENTER, width=25, bg="light sky blue").grid(row=8, column=8)
label7 = Label(master, text="humidity avg", anchor= CENTER, width=25, bg="light sky blue").grid(row=9, column=8)
label8 = Label(master, text="SLPavg", anchor= CENTER, width=25, bg="light sky blue").grid(row=10, column=8)
label9 = Label(master, text="visibilityavg", anchor= CENTER, width=25, bg="light sky blue").grid(row=11, column=8)
label10 = Label(master, text="windavg", anchor= CENTER, width=25, bg="light sky blue").grid(row=12, column=8)
label11 = Label(master, text = "Rainfall", anchor= CENTER, width=25, bg="light sky blue").grid(row=15, column=8)



e1 = Entry(master, bg="honeydew")
e2 = Entry(master, bg="honeydew")
e3 = Entry(master, bg="honeydew")
e4 = Entry(master, bg="honeydew")
e5 = Entry(master, bg="honeydew")
e6 = Entry(master, bg="honeydew")
e7 = Entry(master, bg="honeydew")
e8 = Entry(master, bg="honeydew")
e9 = Entry(master, bg="honeydew")
Rainfall = Entry(master, width=45, bg="honeydew")



e1.grid(row=4, column=16)
e2.grid(row=5, column=16)
e3.grid(row=6, column=16)
e4.grid(row=7, column=16)
e5.grid(row=8, column=16)
e6.grid(row=9, column=16)
e7.grid(row=10, column=16)
e8.grid(row=11, column=16)
e9.grid(row=12, column=16)
Rainfall.grid(row=15, column=16)

##path = "th.jfif"
##
###Create an object of tkinter ImageTk
##img = ImageTk.PhotoImage(Image.open(path))
##
###Create a Label Widget to display the text or Image
##label = tk.Label(master, image = img)
##label.pack(fill = "both", expand = "yes")

Button(master, text='Quit', command=master.destroy,width=10, bg="light steel blue").grid(row=14, column=8, sticky=S, pady=1)
Button(master, text='Find Rainfall', command=showRainfall,width=11, bg="light steel blue").grid(row=14, column=16, sticky=S, pady=1)
contents="Rainfall prediction \n"


mainloop()
    
