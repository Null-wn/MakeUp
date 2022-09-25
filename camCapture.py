import tkinter as tk
from tkinter.constants import LEFT
import tkinter.filedialog
from PIL import Image, ImageTk
import numpy as np
from recommendation import makeup_recommendation,vgg16_face
import cv2
import os
import sys
current_path = sys.path[0]

mark = Image.open(current_path + "\\mirrors\\board1.png")
mark=mark.convert("RGBA")
isCapture=False

def addFrame(img):
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    img=img.convert("RGBA")
    global mark
    mark = mark.resize(img.size)# 贴图大小
    r, g, b, alpha = mark.split()
    img=Image.composite(mark,img,alpha)
    return img

def resize_photo(pil_image):
  ori_w,ori_h = pil_image.size
  dst_scale = 1.0
  ori_scale = float(ori_h) / ori_w #原高宽比
  if ori_scale >= dst_scale:
    #过高
    width = ori_w
    height = int(width*dst_scale)
    x = 0
    y = (ori_h - height) / 3
  else:
    #过宽
    height = ori_h
    width = int(height*dst_scale)
    x = (ori_w - width) / 2
    y = 0
  #裁剪
  box = (x,y,width+x,height+y)
  #这里的参数可以这么认为：从某图的(x,y)坐标开始截，截到(width+x,height+y)坐标
  #所包围的图像，crop方法与php中的imagecopy方法大为不一样
  newIm = pil_image.crop(box)
  pil_image = newIm.resize((224, 224), Image.ANTIALIAS)
  return pil_image

def loadVideo():  
  ret, frame = cap.read()  
  if ret:
    img=addFrame(frame)
    imgtk = ImageTk.PhotoImage(image=img) 
    pan.imgtk = imgtk
    pan.config(image=imgtk)   
    if isCapture:
     # 这里因为颜色空间不一样,导致最开始摄像头获取的画面颜色很奇怪,解决办法,分解成标准呢B G R 再重新组装
      b,g,r = cv2.split(frame) # 分解Opencv里的标准格式B、G、R
      frame = cv2.merge([r,g,b]) # 将BGR格式转化为常用的RGB格式
      img = resize_photo(Image.fromarray(frame))
      img.save(current_path + "\\middle.jpg")
      root.destroy()
  root.after(1, loadVideo)  # 循环函数，每１毫秒

def Capture():
  global isCapture
  isCapture=True

def get_board_path1():
    global mark
    mark = Image.open(current_path + "\\mirrors\\board1.png")
def get_board_path2():
    global mark
    mark = Image.open(current_path + "\\mirrors\\board2.png")
def get_board_path3():
    global mark
    mark = Image.open(current_path + "\\mirrors\\board3.png")
def get_board_path4():
    global mark
    mark = Image.open(current_path + "\\mirrors\\board4.png")
def get_board_path5():
    global mark
    mark = Image.open(current_path + "\\mirrors\\board5.png")
def get_board_path6():
    global mark
    mark = Image.open(current_path + "\\mirrors\\board6.png")
def get_board_path7():
    global mark
    mark = Image.open(current_path + "\\mirrors\\board7.png")

def show():
  os.system('python showmakeup/showboard.py')


cap = cv2.VideoCapture(0)  # 捕获摄像头

root = tk.Tk()  # 创建一个窗口
root.title('摄像头拍摄') 

frame = tk.Frame(root)
frame.pack(side='top')
frameCH=tk.Frame(root)
frameCH.pack(side='top')

pan = tk.Label(frame, padx=10, pady=10)  # 继承于frame组件，用来存放图像
pan.pack()

btnCap = tk.Button(frame, text="    拍照    ",command=Capture)  # 创建按钮对象，继承于frame，并设置按钮的文本内容
btnCap.pack(side='left')
btnChoose = tk.Button(frame,text="镜框预览",command=show)
btnChoose.pack(side='left')
var1 = tk.StringVar()
r7 = tk.Radiobutton(frameCH, text='镜框1', variable=var1, value='镜框1',command = get_board_path1)
r7.pack(side='left')
r8 = tk.Radiobutton(frameCH, text='镜框2', variable=var1, value='镜框2',command = get_board_path2)
r8.pack(side='left')
r9 = tk.Radiobutton(frameCH, text='镜框3', variable=var1, value='镜框3',command = get_board_path3)
r9.pack(side='left')
r10 = tk.Radiobutton(frameCH, text='镜框4', variable=var1, value='镜框4',command = get_board_path4)
r10.pack(side='left')
r11 = tk.Radiobutton(frameCH, text='镜框5', variable=var1, value='镜框5',command = get_board_path5)
r11.pack(side='left')
r12 = tk.Radiobutton(frameCH, text='镜框6', variable=var1, value='镜框6',command = get_board_path6)
r12.pack(side='left')
r13 = tk.Radiobutton(frameCH, text='镜框7', variable=var1, value='镜框7',command = get_board_path7)
r13.pack(side='left')
# 函数循环，事件绑定
loadVideo()
root.mainloop()
cap.release() 
cv2.destroyAllWindows()

