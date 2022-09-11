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
root2.title("欧美展示")
root2.geometry("1090x230+300+150")
button1 = tk.Button(root2, text='开始选择',command=quit_window)
button1.pack()
imgs = ['imgs/America1/1.jpg', 'imgs/America1/2.jpg', 'imgs/America1/3.jpg', 'imgs/America1/4.jpg', 'imgs/America1/5.jpg', 'imgs/America1/6.jpg', 'imgs/America1/7.jpg', 'imgs/America1/8.jpg','imgs/America1/9.jpg']
for i in range(len(imgs)):
    load_img(i+1,imgs[i])
root2.mainloop()

