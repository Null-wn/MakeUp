import tkinter as tk
from PIL import Image, ImageTk

def quit_window():
    root2.destroy()

def load_img(index,path):
    paned = tk.PanedWindow(root2)
    paned.pack(fill=tk.X, side=tk.LEFT)
    img = Image.open(path)
    paned.photo = ImageTk.PhotoImage(img.resize((115, 137)))
    tk.Label(paned, image=paned.photo).grid(row=index, column=0)

root2 = tk.Tk()
root2.title("日常妆容")
root2.geometry("1090x230+300+150")
button1 = tk.Button(root2, text='开始选择',command=quit_window)
button1.pack()
imgs = ['imgs/daily1/1.jpg', 'imgs/daily1/2.jpg', 'imgs/daily1/3.jpg', 'imgs/daily1/4.jpg', 'imgs/daily1/5.jpg', 'imgs/daily1/6.jpg', 'imgs/daily1/7.jpg', 'imgs/daily1/8.jpg', 'imgs/daily1/9.jpg']
for i in range(len(imgs)):
    load_img(i+1,imgs[i])
root2.mainloop()
