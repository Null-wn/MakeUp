import tkinter as tk
from tkinter.constants import FALSE, FLAT
import tkinter.filedialog
from PIL import Image, ImageTk, ImageFilter
import numpy as np
import cv2
import cv2 as cv
from scipy.interpolate import CubicSpline


def Showimage(img1,img2,canva1,canva2):
    global imgTK1
    canvawidth = int(canva1.winfo_reqwidth())
    canvaheight = int(canva1.winfo_reqheight())
    sp = img1.shape
    cvheight = sp[0]#height(rows) of image
    cvwidth = sp[1]#width(colums) of image
    if (float(cvwidth/cvheight) > float(canvawidth/canvaheight)):
        imgCV = cv.resize(img1,(canvawidth,int(canvawidth*cvheight/cvwidth)), interpolation=cv.INTER_AREA)
    else:
        imgCV = cv.resize(img1,(int(canvaheight*cvwidth/cvheight),canvaheight), interpolation=cv.INTER_AREA)
    imgCV2 = cv.cvtColor(imgCV, cv.COLOR_BGR2RGBA)#转换颜色从BGR到RGBA
    current_image = Image.fromarray(imgCV2)#将图像转换成Image对象
    imgTK1 = ImageTk.PhotoImage(image=current_image)#将image对象转换为imageTK对象
    canva1.create_image(150,150,image = imgTK1,anchor='center')

    global imgTK2
    canvawidth = int(canva2.winfo_reqwidth())
    canvaheight = int(canva2.winfo_reqheight())
    sp = img2.shape
    cvheight = sp[0]#height(rows) of image
    cvwidth = sp[1]#width(colums) of image
    if (float(cvwidth/cvheight) > float(canvawidth/canvaheight)):
        imgCV = cv.resize(img2,(canvawidth,int(canvawidth*cvheight/cvwidth)), interpolation=cv.INTER_AREA)
    else:
        imgCV = cv.resize(img2,(int(canvaheight*cvwidth/cvheight),canvaheight), interpolation=cv.INTER_AREA)
    imgCV2 = cv.cvtColor(imgCV, cv.COLOR_BGR2RGBA)#转换颜色从BGR到RGBA
    current_image = Image.fromarray(imgCV2)#将图像转换成Image对象
    imgTK2 = ImageTk.PhotoImage(image=current_image)#将image对象转换为imageTK对象
    canva2.create_image(150,150,image = imgTK2,anchor='center')

def skinSmoothing2(img):
    def np2pil(numpy_image):
        return Image.fromarray(np.uint8(numpy_image*255.0)).convert('RGB')

    input_img = np.array(img/255.0,dtype=np.float32) #rgb
    ea_img = input_img * pow(2,-1.0)
    base = ea_img[...,1]
    overlay = ea_img[...,2]
    ba = 2.0*overlay*base
    ba_img = np.zeros((ba.shape[0],ba.shape[1],3),dtype=np.float32)
    ba_img[...,0] = ba
    ba_img[...,1] = ba
    ba_img[...,2] = ba
    radius = int(np.ceil(7.0*input_img.shape[0]/750.0))
    pil_img = np2pil(ba_img)
    pil_blur = pil_img.filter(ImageFilter.GaussianBlur(radius))
    blur_img = np.asarray(pil_blur,np.float32)/255.0
    hp_img = ba_img - blur_img + 0.5
    hardLightColor = hp_img[...,2]
    [x1,y1] = np.where(hardLightColor<0.5)
    [x2,y2] = np.where(hardLightColor>=0.5)
    for i in range(3):
        hardLightColor[x1,y1] = hardLightColor[x1,y1]*hardLightColor[x1,y1]*2.0
        hardLightColor[x2,y2] = 1.0 - (1.0 - hardLightColor[x2,y2]) * (1.0 - hardLightColor[x2,y2]) * 2.0
    k = 255.0/(164.0-75.0);
    hardLightColor = (hardLightColor - 75.0/255.0) * k
    hpss_img = np.zeros((hardLightColor.shape[0],hardLightColor.shape[1],3))
    hpss_img[...,0] = hardLightColor
    hpss_img[...,1] = hardLightColor
    hpss_img[...,2] = hardLightColor
    hpss_img = np.clip(hpss_img,0,1)
    x = [0,120.0/255.0,1]
    y = [0,146.0/255.0,1]#146
    cs = CubicSpline(x,y)
    tc_img = cs(input_img)
    blend_img = input_img * hpss_img + tc_img*(1-hpss_img)
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Sharpness(np2pil(blend_img))
    img_sharp = enhancer.enhance(2)
    result = np.array(img_sharp,np.float32)/255.0
    return result

def SaveAs():
    img=dst
    path = tk.filedialog.asksaveasfilename()
    cv2.imwrite(path + ".jpg", img)
    if path:
        tk.messagebox.showinfo(title='保存成功', message="已成功保存到：" + path + ".jpg")
    else:
        tk.messagebox.showwarning(title='保存失败', message="保存失败，请检查保存路径")

root = tk.Tk()
root.geometry('640x410+600+300')
root.title("深度磨皮")
canva1 = tk.Canvas(root, width=300, height=300,bg="white")
canva1.place(x=5,y=45)
canva2 = tk.Canvas(root, width=300, height=300,bg="white")
canva2.place(x=330,y=45)
label1 = tk.Label(root, text='Source', font=('Arial Black', 12), width=15, height=1 )
label1.place(x=155,y=15,anchor='n')
label2 = tk.Label(root, text='Destination', font=('Arial Black', 12), width=15, height=1 )
label2.place(x=480,y=15,anchor='n')
btn_SaveAs = tk.Button(root, text="保存", padx=20, pady=1.5, font=('Arial', 12), command=SaveAs)  # 保存
btn_SaveAs.place(x=320, y=380, anchor="center")
path = tk.filedialog.askopenfilename()
src=cv2.imread(path)
dst = skinSmoothing2(src)
Showimage(src,dst,canva1,canva2)
root.mainloop()
