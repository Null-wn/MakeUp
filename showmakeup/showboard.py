import tkinter as tk
from PIL import Image, ImageTk

def quit_window():
    root2.destroy()

def load_img(index,path):
    paned = tk.PanedWindow(root2)
    paned.pack(fill=tk.X, side=tk.LEFT)
    img = Image.open(path)
    paned.photo = ImageTk.PhotoImage(img.resize((150, 210)))
    tk.Label(paned, image=paned.photo).grid(row=index, column=0)

root2 = tk.Tk()
root2.title("镜框展示")
root2.geometry("1100x300+300+150")
button1 = tk.Button(root2, text='开始选择',command=quit_window)
button1.pack()
mirrors = ['mirrors/board1.png', 'mirrors/board2.png', 'mirrors/board3.png', 'mirrors/board4.png', 'mirrors/board5.png','mirrors/board6.png', 'mirrors/board7.png']
for i in range(len(mirrors)):
    load_img(i+1,mirrors[i])
root2.mainloop()
