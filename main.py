import tkinter as tk
from tkinter.constants import FALSE, FLAT
import tkinter.filedialog
import tkinter.messagebox
from PIL import Image, ImageTk
from torch._C import Size, wait
import makeupTransfer
import numpy as np
from recommendation import makeup_recommendation, vgg16_face
import cv2
import cv2 as cv
import os
import sys

current_path = sys.path[0]
mark = Image.open(current_path + "\\mirrors\\board5.png")


class Modal(tk.Toplevel):
    """
    Modal about dialog for this pragram.
    """

    def __init__(self, parent, title=None):
        """
        Create popup, do not return until widget destroyed. 创建弹出窗口，不要返回，直到小部件销毁。
        parent - parent of this dialog 这个对话框的父节点
        title - string which is title of popup dialog 标题-字符串，这是弹出对话框的标题
        """
        tk.Toplevel.__init__(self, parent, width=200, height=300)
        self.geometry("400x330+%d+%d" % (parent.winfo_rootx() + 120, parent.winfo_rooty()))
        self.title(title)
        self.resizable(height=False, width=False)
        self.grab_set()
        self.initUI()
        btn_SaveAs = tk.Button(self, text="保存", padx=10, pady=2, bd=3, command=SaveAs)  # 保存
        btn_SaveAs.place(x=200, y=305, anchor="center")

    def initUI(self):
        self.frame = tk.Frame(self)
        self.frame.pack(fill='x', expand=True)

        img_ = resize_pic('result.jpg')  # 上妆后的结果保存的名字
        lb_result = tk.Label(self.frame, image=img_, relief="raised", bd=4)
        lb_result.image = img_
        lb_result.pack(padx=5, expand=True)

def SaveAs():
    img = cv2.imread('result.jpg', 1)  # 上妆后的结果保存的名字
    # cv2.imshow("show",img)
    path = tk.filedialog.asksaveasfilename()
    cv2.imwrite(path + ".jpg", img)
    if path:
        tkinter.messagebox.showinfo(title='保存成功', message="已成功保存到：" + path + ".jpg")
    else:
        tkinter.messagebox.showwarning(title='保存失败', message="保存失败，请检查保存路径")

def AgeGenderDetect(path):  
    global root
    def getFaceBox(net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                if  x1<frameWidth and x1>0 and x2<frameWidth and x2>0 and y1<frameHeight and y1>0 and y2<frameHeight and y2>0:
                    bboxes.append([x1, y1, x2, y2])
                    cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)),8)
        return frameOpencvDnn,bboxes

    faceProto = current_path +"\\opencv_tutorial\\data\\models\\face_detector\\opencv_face_detector.pbtxt"
    faceModel = current_path +"\\opencv_tutorial\\data\\models\\face_detector\\opencv_face_detector_uint8.pb"
    ageProto = current_path + "\\opencv_tutorial\\data\\models\\cnn_age_gender_models\\age_deploy.prototxt"
    ageModel = current_path + "\\opencv_tutorial\\data\\models\\cnn_age_gender_models\\age_net.caffemodel"
    genderProto = current_path + "\\opencv_tutorial\\data\\models\\cnn_age_gender_models\\gender_deploy.prototxt"
    genderModel = current_path + "\\opencv_tutorial\\data\\models\\cnn_age_gender_models\\gender_net.caffemodel"
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']
    # Load network
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    faceNet = cv2.dnn.readNet(faceModel, faceProto)

    img=cv2.imread(path)
    bounding,bboxes = getFaceBox(faceNet,img)

    if not bboxes :
        tk.messagebox.showwarning(title="检测失败",message="检测不到人脸！")

    else :
        for bbox in bboxes:
            face = img[max(0,bbox[1]-20):min(bbox[3]+20,img.shape[0]-1),max(0,bbox[0]-20):min(bbox[2]+20, img.shape[1]-1)]      
            #while(1):
                #cv2.imshow("?",bounding)
                #if cv2.waitKey(0)==27:
                #    break;
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
        var=tkinter.StringVar()
        var.set("性别 : {}".format(gender)+"    年龄 : {}".format(age))
        SA_Label=tk.Label(root,textvariable=var)
        SA_Label.place(x=80,y=250)
        print('var')

def resize_pic(img_path):
    pil_image = Image.open(img_path)
    img = pil_image.resize((224, 224), Image.ANTIALIAS)
    tkImage = ImageTk.PhotoImage(image=img)
    return tkImage


def resize_pic1(img_path):
    pil_image = Image.open(img_path)
    img = pil_image.resize((120, 120), Image.ANTIALIAS)
    tkImage = ImageTk.PhotoImage(image=img)
    return tkImage


def select_path():
    # 选择文件path_接收文件地址
    path_ = tk.filedialog.askopenfilename()

    # 通过replace函数替换绝对文件地址中的/来使文件可被程序读取
    # 注意：\\转义后为\，所以\\\\转义后为\\
    path_ = path_.replace("/", "\\\\")
    return path_


# 登陆注册
def bt3_1_callback():
    os.system('python example.py')


# 选择素颜照片
def bt1_callback():
    path_ = select_path()
    global before_makeup_path
    before_makeup_path = path_
    img_ = resize_pic(path_)
    label1.config(image=img_)
    label1.image = img_  # keep a reference
    AgeGenderDetect(before_makeup_path)



# 推荐妆容
def bt2_callback():
    global before_makeup_path
    result = makeup_recommendation(before_makeup_path)
    recommendation_path = os.path.join('imgs', 'recommendation', result + '.jpg')
    global coll
    coll = result + '.jpg'
    global after_makeup_path
    after_makeup_path = recommendation_path
    img_ = resize_pic(recommendation_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    star=resize_pic3(current_path + "\\imgs\\shoucang1.jpeg")
    bt11.config(image=star)
    bt11.image=star


# 自选妆容照片、收藏照片
def bt3_callback():
    path_ = select_path()
    global after_makeup_path
    after_makeup_path = path_
    img_ = resize_pic(path_)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    a = os.path.split(path_)
    global coll
    coll = a[1]
    star=resize_pic3(current_path + "\\imgs\\shoucang1.jpeg")
    bt11.config(image=star)
    bt11.image=star


def open_image_by_path(path_1):
    global before_makeup_path
    before_makeup_path = path_1
    img_ = resize_pic(path_1)
    label1.config(image=img_)
    label1.image = img_  # keep a reference
    AgeGenderDetect(before_makeup_path)



def addFrame(img, x, y, w, h):
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    global mark
    mark = mark.resize((244 * 3, 244 * 3))
    layer = Image.new('RGBA', im.size, (0, 0, 0, 0))
    layer.paste(mark, (x, y))
    out = Image.composite(layer, im, layer)
    img = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
    return img


# 上妆
def bt4_callback():
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")


# 收藏
def bt11_callback():
    global coll
    img = cv2.imread(after_makeup_path, 1)
    path = current_path + "/imgs/collection/" + coll
    cv2.imwrite(path, img)
    star=resize_pic3(current_path + "\\imgs\\shoucang2.jpeg")
    bt11.config(image=star)
    bt11.image=star
    if path:
        tk.messagebox.showinfo(title='收藏成功', message="已成功收藏到：" + path + ".jpg")
    else:
        tk.messagebox.showwarning(title='收藏失败', message="收藏失败，请检查收藏路径")
    cv2.waitKey(0)
   


def resize_pic(img_path):
    pil_image = Image.open(img_path)
    img = pil_image.resize((224, 224), Image.ANTIALIAS)
    tkImage = ImageTk.PhotoImage(image=img)
    return tkImage

def resize_pic1(img_path):
    pil_image = Image.open(img_path)
    img = pil_image.resize((120, 120), Image.ANTIALIAS)
    tkImage = ImageTk.PhotoImage(image=img)
    return tkImage

def resize_pic2(img_path):
    pil_image = Image.open(img_path)
    img = pil_image.resize((670, 560), Image.ANTIALIAS)
    tkImage = ImageTk.PhotoImage(image=img)
    return tkImage

def resize_pic3(img_path):
    pil_image = Image.open(img_path)
    img = pil_image.resize((40, 40), Image.ANTIALIAS)
    tkImage = ImageTk.PhotoImage(image=img)
    return tkImage

def resize_photo(pil_image):
    ori_w, ori_h = pil_image.size
    dst_scale = 1.0
    ori_scale = float(ori_h) / ori_w  # 原高宽比
    if ori_scale >= dst_scale:
        # 过高
        width = ori_w
        height = int(width * dst_scale)
        x = 0
        y = (ori_h - height) / 3
    else:
        # 过宽
        height = ori_h
        width = int(height * dst_scale)
        x = (ori_w - width) / 2
        y = 0
    # 裁剪
    box = (x, y, width + x, height + y)
    # 这里的参数可以这么认为：从某图的(x,y)坐标开始截，截到(width+x,height+y)坐标
    # 所包围的图像，crop方法与php中的imagecopy方法大为不一样
    newIm = pil_image.crop(box)
    pil_image = newIm.resize((224, 224), Image.ANTIALIAS)
    return pil_image


# 摄像功能
def bt1_1_callback():
    os.system('python camCapture.py')
    open_image_by_path(current_path + "\\middle.jpg")

def skinSmoothing_callback():
    os.system('python skinSmoothing1.py')
    
#def skinSmoothing2_callback():
#    os.system('python skinSmoothing2.py')

# 日常妆容，showmakeup1
def bt5_callback():
    for i in range(9):
        img[i] = resize_pic1(current_path + "\\imgs\\daily1\\" + str(i + 1) + ".jpg")
        r[i].config(image=img[i], command=get_daily1_path)
        r[i].image = img[i]
        r[i].command = get_daily1_path


# 获取日常妆容，并上妆
def get_daily1_path():
    global after_makeup_path
    global coll
    global var
    num = var.get()
    coll = 'daily1' + str(num) + '.jpg'
    after_makeup_path = 'imgs/daily1/' + str(num) + '.jpg'
    img_ = resize_pic(after_makeup_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")
    star=resize_pic3(current_path + "\\imgs\\shoucang1.jpeg")
    bt11.config(image=star)
    bt11.image=star


def bt6_callback():
    for i in range(9):
        img[i] = resize_pic1(current_path + "\\imgs\\party1\\" + str(i + 1) + ".jpg")
        r[i].config(image=img[i], command=get_party1_path)
        r[i].image = img[i]
        r[i].command = get_daily1_path


def get_party1_path():
    global after_makeup_path
    global coll
    global var
    num = var.get()
    coll = 'party1' + str(num) + '.jpg'
    after_makeup_path = 'imgs/party1/' + str(num) + '.jpg'
    img_ = resize_pic(after_makeup_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")
    star=resize_pic3(current_path + "\\imgs\\shoucang1.jpeg")
    bt11.config(image=star)
    bt11.image=star


def bt7_callback():
    for i in range(9):
        img[i] = resize_pic1(current_path + "\\imgs\\work1\\" + str(i + 1) + ".jpg")
        r[i].config(image=img[i], command=get_work1_path)
        r[i].image = img[i]
        r[i].command = get_daily1_path


def get_work1_path():
    global after_makeup_path
    global coll
    global var
    num = var.get()
    coll = 'work1' + str(num) + '.jpg'
    after_makeup_path = 'imgs/work1/' + str(num) + '.jpg'
    img_ = resize_pic(after_makeup_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")
    star=resize_pic3(current_path + "\\imgs\\shoucang1.jpeg")
    bt11.config(image=star)
    bt11.image=star


def bt8_callback():
    for i in range(9):
        img[i] = resize_pic1(current_path + "\\imgs\\China1\\" + str(i + 1) + ".jpg")
        r[i].config(image=img[i], command=get_China1_path)
        r[i].image = img[i]
        r[i].command = get_daily1_path


def get_China1_path():
    global after_makeup_path
    global coll
    global var
    num = var.get()
    coll = 'China1' + str(num) + '.jpg'
    after_makeup_path = 'imgs/China1/' + str(num) + '.jpg'
    img_ = resize_pic(after_makeup_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")
    star=resize_pic3(current_path + "\\imgs\\shoucang1.jpeg")
    bt11.config(image=star)
    bt11.image=star


def bt9_callback():
    for i in range(9):
        img[i] = resize_pic1(current_path + "\\imgs\\Korea1\\" + str(i + 1) + ".jpg")
        r[i].config(image=img[i], command=get_Korea1_path)
        r[i].image = img[i]
        r[i].command = get_daily1_path


def get_Korea1_path():
    global after_makeup_path
    global coll
    global var
    num = var.get()
    coll = 'Korea1' + str(num) + '.jpg'
    after_makeup_path = 'imgs/Korea1/' + str(num) + '.jpg'
    img_ = resize_pic(after_makeup_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")
    star=resize_pic3(current_path + "\\imgs\\shoucang1.jpeg")
    bt11.config(image=star)
    bt11.image=star


def bt10_callback():
    for i in range(9):
        img[i] = resize_pic1(current_path + "\\imgs\\America1\\" + str(i + 1) + ".jpg")
        r[i].config(image=img[i], command=get_America1_path)
        r[i].image = img[i]
        r[i].command = get_daily1_path

def get_America1_path():
    global after_makeup_path
    global coll
    global var
    num = var.get()
    coll = 'America1' + str(num) + '.jpg'
    after_makeup_path = 'imgs/America1/' + str(num) + '.jpg'
    img_ = resize_pic(after_makeup_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")
    star=resize_pic3(current_path + "\\imgs\\shoucang1.jpeg")
    bt11.config(image=star)
    bt11.image=star

# 日常妆容，showmakeup1
def bt5_1_callback():
    for i in range(9):
        img[i] = resize_pic1(current_path + "\\imgs\\daily2\\" + str(i + 1) + ".jpg")
        r[i].config(image=img[i], command=get_daily2_path)
        r[i].image = img[i]
        r[i].command = get_daily2_path

# 获取日常妆容，并上妆
def get_daily2_path():
    global after_makeup_path
    global coll
    global var
    num = var.get()
    coll = 'daily2' + str(num) + '.jpg'
    after_makeup_path = 'imgs/daily2/' + str(num) + '.jpg'
    img_ = resize_pic(after_makeup_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")
    star=resize_pic3(current_path + "\\imgs\\shoucang1.jpeg")
    bt11.config(image=star)
    bt11.image=star

def bt6_1_callback():
    for i in range(9):
        img[i] = resize_pic1(current_path + "\\imgs\\party2\\" + str(i + 1) + ".jpg")
        r[i].config(image=img[i], command=get_party2_path)
        r[i].image = img[i]
        r[i].command = get_daily2_path


def get_party2_path():
    global after_makeup_path
    global coll
    global var
    num = var.get()
    coll = 'party2' + str(num) + '.jpg'
    after_makeup_path = 'imgs/party2/' + str(num) + '.jpg'
    img_ = resize_pic(after_makeup_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")
    star=resize_pic3(current_path + "\\imgs\\shoucang1.jpeg")
    bt11.config(image=star)
    bt11.image=star


def bt7_1_callback():
    for i in range(9):
        img[i] = resize_pic1(current_path + "\\imgs\\work2\\" + str(i + 1) + ".jpg")
        r[i].config(image=img[i], command=get_work2_path)
        r[i].image = img[i]
        r[i].command = get_daily2_path


def get_work2_path():
    global after_makeup_path
    global coll
    global var
    num = var.get()
    coll = 'work2' + str(num) + '.jpg'
    after_makeup_path = 'imgs/work2/' + str(num) + '.jpg'
    img_ = resize_pic(after_makeup_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")
    star=resize_pic3(current_path + "\\imgs\\shoucang1.jpeg")
    bt11.config(image=star)
    bt11.image=star


def bt8_1_callback():
    for i in range(9):
        img[i] = resize_pic1(current_path + "\\imgs\\China2\\" + str(i + 1) + ".jpg")
        r[i].config(image=img[i], command=get_China2_path)
        r[i].image = img[i]
        r[i].command = get_daily2_path


def get_China2_path():
    global after_makeup_path
    global coll
    global var
    num = var.get()
    coll = 'China2' + str(num) + '.jpg'
    after_makeup_path = 'imgs/China2/' + str(num) + '.jpg'
    img_ = resize_pic(after_makeup_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")
    star=resize_pic3(current_path + "\\imgs\\shoucang1.jpeg")
    bt11.config(image=star)
    bt11.image=star


def bt9_1_callback():
    for i in range(9):
        img[i] = resize_pic1(current_path + "\\imgs\\Korea2\\" + str(i + 1) + ".jpg")
        r[i].config(image=img[i], command=get_Korea2_path)
        r[i].image = img[i]
        r[i].command = get_daily2_path


def get_Korea2_path():
    global after_makeup_path
    global coll
    global var
    num = var.get()
    coll = 'Korea2' + str(num) + '.jpg'
    after_makeup_path = 'imgs/Korea2/' + str(num) + '.jpg'
    img_ = resize_pic(after_makeup_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")
    star=resize_pic3(current_path + "\\imgs\\shoucang1.jpeg")
    bt11.config(image=star)
    bt11.image=star


def bt10_1_callback():
    for i in range(9):
        img[i] = resize_pic1(current_path + "\\imgs\\America2\\" + str(i + 1) + ".jpg")
        r[i].config(image=img[i], command=get_America2_path)
        r[i].image = img[i]
        r[i].command = get_daily2_path


def get_America2_path():
    global after_makeup_path
    global coll
    global var
    num = var.get()
    coll = 'America2' + str(num) + '.jpg'
    after_makeup_path = 'imgs/America2/' + str(num) + '.jpg'
    img_ = resize_pic(after_makeup_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")
    star=resize_pic3(current_path + "\\imgs\\shoucang1.jpeg")
    bt11.config(image=star)
    bt11.image=star


def bt14_callback():
    for i in range(9):
        img[i] = resize_pic1(current_path + "\\ages\\first\\" + str(i + 1) + ".jpg")
        r[i].config(image=img[i], command=get_ages1_path)
        r[i].image = img[i]
        r[i].command = get_ages1_path


def get_ages1_path():
    global after_makeup_path
    global coll
    global var
    num = var.get()
    coll = 'first' + str(num) + '.jpg'
    after_makeup_path = 'ages/first/' + str(num) + '.jpg'
    img_ = resize_pic(after_makeup_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")

def bt16_callback():
    for i in range(9):
        img[i] = resize_pic1(current_path + "\\ages\\fourth\\" + str(i + 1) + ".jpg")
        r[i].config(image=img[i], command=get_ages4_path)
        r[i].image = img[i]
        r[i].command = get_ages4_path


def get_ages4_path():
    global after_makeup_path
    global coll
    global var
    num = var.get()
    coll = 'fourth' + str(num) + '.jpg'
    after_makeup_path = 'ages/fourth/' + str(num) + '.jpg'
    img_ = resize_pic(after_makeup_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")

def bt12_callback():
    for i in range(9):
        img[i] = resize_pic1(current_path + "\\ages\\second\\" + str(i + 1) + ".jpg")
        r[i].config(image=img[i], command=get_ages2_path)
        r[i].image = img[i]
        r[i].command = get_ages2_path


def get_ages2_path():
    global after_makeup_path
    global coll
    global var
    num = var.get()
    coll = 'second' + str(num) + '.jpg'
    after_makeup_path = 'ages/first/' + str(num) + '.jpg'
    img_ = resize_pic(after_makeup_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")

def bt17_callback():
    for i in range(9):
        img[i] = resize_pic1(current_path + "\\ages\\fifth\\" + str(i + 1) + ".jpg")
        r[i].config(image=img[i], command=get_ages5_path)
        r[i].image = img[i]
        r[i].command = get_ages5_path

def get_ages5_path():
    global after_makeup_path
    global coll
    global var
    num = var.get()
    coll = 'fifth' + str(num) + '.jpg'
    after_makeup_path = 'ages/fifth/' + str(num) + '.jpg'
    img_ = resize_pic(after_makeup_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")

def bt13_callback():
    for i in range(9):
        img[i] = resize_pic1(current_path + "\\ages\\third\\" + str(i + 1) + ".jpg")
        r[i].config(image=img[i], command=get_ages3_path)
        r[i].image = img[i]
        r[i].command = get_ages3_path

def get_ages3_path():
    global after_makeup_path
    global coll
    global var
    num = var.get()
    coll = 'third' + str(num) + '.jpg'
    after_makeup_path = 'ages/third/' + str(num) + '.jpg'
    img_ = resize_pic(after_makeup_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")

def bt18_callback():
    for i in range(9):
        img[i] = resize_pic1(current_path + "\\ages\\sixth\\" + str(i + 1) + ".jpg")
        r[i].config(image=img[i], command=get_ages6_path)
        r[i].image = img[i]
        r[i].command = get_ages6_path


def get_ages6_path():
    global after_makeup_path
    global coll
    global var
    num = var.get()
    coll = 'sixth' + str(num) + '.jpg'
    after_makeup_path = 'ages/sixth/' + str(num) + '.jpg'
    img_ = resize_pic(after_makeup_path)
    label2.config(image=img_)
    label2.image = img_  # keep a reference
    makeupTransfer.makeup_transfer(before_makeup_path, after_makeup_path)
    Modal(root, title="上妆结果")

def bt15_callback():
    os.system("start explorer D:\dachuang\makeup\imgs\collection")

def bt19_callback():
    image = cv.imread("result.jpg",cv.IMREAD_GRAYSCALE)
    binary = cv.adaptiveThreshold(image,255,
                              cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,25,15)
    se = cv.getStructuringElement(cv.MORPH_RECT,(1,1))
    se = cv.morphologyEx(se, cv.MORPH_CLOSE, (2,2))
    mask = cv.dilate(binary,se)

    mask1 = cv.bitwise_not(mask)
    binary =cv.bitwise_and(image,mask)
    result = cv.add(binary,mask1)
    cv.imshow("reslut",result)
    cv.imwrite("reslut1.jpg",result)
    cv.waitKey(0)
    cv.destroyAllWindows()

def bt20_callback():
    image = cv.imread(before_makeup_path,cv.IMREAD_GRAYSCALE)
    binary = cv.adaptiveThreshold(image,255,
                              cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,25,15)
    se = cv.getStructuringElement(cv.MORPH_RECT,(1,1))
    se = cv.morphologyEx(se, cv.MORPH_CLOSE, (2,2))
    mask = cv.dilate(binary,se)

    mask1 = cv.bitwise_not(mask)
    binary =cv.bitwise_and(image,mask)
    result = cv.add(binary,mask1)
    cv.imshow("reslut",result)
    cv.imwrite("reslut2.jpg",result)
    cv.waitKey(0)
    cv.destroyAllWindows()

def bt21_callback():
    path_ = select_path()
    image = cv.imread(path_,cv.IMREAD_GRAYSCALE)
    binary = cv.adaptiveThreshold(image,255,
                              cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,25,15)
    se = cv.getStructuringElement(cv.MORPH_RECT,(1,1))
    se = cv.morphologyEx(se, cv.MORPH_CLOSE, (2,2))
    mask = cv.dilate(binary,se)

    mask1 = cv.bitwise_not(mask)
    binary =cv.bitwise_and(image,mask)
    result = cv.add(binary,mask1)
    cv.imshow("reslut",result)
    cv.imwrite("reslut2.jpg",result)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def set_SexAge():
    rootSet=tk.Tk()
    rootSet.title("设置年龄和性别")
    rootSet.geometry("320x180+450+250")
    rootSet.resizable(width=FALSE,height=FALSE)
    tk.Label(rootSet, text='性别: ').place(x=30, y=25)
    tk.Label(rootSet, text='年龄: ').place(x=30, y=85)
    var = tk.StringVar()    # 定义一个var用来将radiobutton的值和Label的值联系在一起.
    def done():
        if sex=="Male":
            print("Male")#后续的根据性别的变化的可以在这里改
        elif sex=="Female":
            print("Female")#后续的根据性别的变化的可以在这里改
        else:
            print("error")
    
        if age<30 and age>17:
            #bt14_callback()
            print("<30")
        elif age>30 and age<50:
            #bt12_callback()
            print("30~50")
        elif age>50:
            #bt13_callback()
            print("50+")
        else :
            print("error")
        tk.messagebox.showinfo(title='设置成功', message="Sex: "+sex+"   "+"Age: "+str(age))
        rootSet.destroy()

    def setMale():
        global sex
        sex="Male"
        #print(sex)

    def setFemale():
        global sex
        sex="Female"
        #print(sex)

    def setAge(v):
        global age
        age=int(v)
        #print(age)
 
    tk.Button(rootSet, text='Male', command=setMale).place(x=90,y=25)
    tk.Button(rootSet, text='Female', command=setFemale).place(x=195,y=25)
    tk.Scale(rootSet, from_=18, to=85, orient=tk.HORIZONTAL, length=200, showvalue=1,command=setAge).place(x=75,y=68)
    tk.Button(rootSet,text='确定',padx=10, pady=2, bd=3, command=done).place(x=160,y=150,anchor="center")


root = tk.Tk()  # 创建主窗口root
root.title("人脸妆容推荐系统")  # 给可视化窗口起名字
root.geometry("670x560+300+150")  # 设定窗口的大小(长 * 宽)

# 创建菜单
menubar = tk.Menu(root)
# 菜单一
filemenu = tk.Menu(menubar, tearoff=0)
# 一级菜单
menubar.add_cascade(label='打开图片', menu=filemenu)
# 二级菜单
filemenu.add_command(label='从相册中选择', command=bt1_callback)
filemenu.add_command(label='拍摄', command=bt1_1_callback)
# 菜单二
editmenu = tk.Menu(menubar, tearoff=0)
# 一级菜单
menubar.add_cascade(label='选择妆容', menu=editmenu)
# 二级菜单
editmenu.add_command(label='妆容推荐', command=bt2_callback)
editmenu.add_command(label='自选妆容', command=bt3_callback)
#
# 菜单一子菜单
submenu1 = tk.Menu(editmenu)
submenu = tk.Menu(submenu1)
submenu2 = tk.Menu(submenu1)
# 一级菜单
editmenu.add_cascade(label='选择妆容风格', menu=submenu1)
submenu1.add_cascade(label='男', menu=submenu2)
submenu1.add_cascade(label='女', menu=submenu)
# 二级菜单
submenu.add_command(label='日常妆容', command=bt5_callback)
submenu.add_command(label='聚会妆容', command=bt6_callback)
submenu.add_command(label='通勤妆容', command=bt7_callback)
submenu.add_command(label='国风妆容', command=bt8_callback)
submenu.add_command(label='日韩妆容', command=bt9_callback)
submenu.add_command(label='欧美妆容', command=bt10_callback)
submenu2.add_command(label='日常妆容', command=bt5_1_callback)
submenu2.add_command(label='聚会妆容', command=bt6_1_callback)
submenu2.add_command(label='通勤妆容', command=bt7_1_callback)
submenu2.add_command(label='国风妆容', command=bt8_1_callback)
submenu2.add_command(label='日韩妆容', command=bt9_1_callback)
submenu2.add_command(label='欧美妆容', command=bt10_1_callback)
# 菜单一子菜单
#submenu = tk.Menu(editmenu)
# 一级菜单
#editmenu.add_cascade(label='选择年龄', menu=submenu)
# 二级菜单
#submenu.add_command(label='18-30', command=bt14_callback)
#submenu.add_command(label='30-50', command=bt12_callback)
#submenu.add_command(label='50+', command=bt13_callback)
# 菜单一子菜单
submenu = tk.Menu(editmenu)
# 一级菜单
editmenu.add_cascade(label='选择年龄', menu=submenu)
# 二级菜单
filemenu1 = tk.Menu(submenu)
filemenu2 = tk.Menu(submenu)
submenu.add_cascade(label='男', menu=filemenu1)
submenu.add_cascade(label='女', menu=filemenu2)
filemenu1.add_command(label='18-30', command=bt16_callback)
filemenu1.add_command(label='30-50', command=bt17_callback)
filemenu1.add_command(label='50+', command=bt18_callback)
filemenu2.add_command(label='18-30', command=bt14_callback)
filemenu2.add_command(label='30-50', command=bt12_callback)
filemenu2.add_command(label='50+', command=bt13_callback)
# 菜单三
editmenu = tk.Menu(menubar, tearoff=0)
# 一级菜单
menubar.add_cascade(label='登录注册', menu=editmenu)
# 二级菜单
editmenu.add_command(label='登录注册', command=bt3_1_callback)
#editmenu.add_command(label='年龄性别',command=set_SexAge)
# 菜单四
editmenu = tk.Menu(menubar, tearoff=0)
# 一级菜单
menubar.add_cascade(label='我的收藏', menu=editmenu)
# 二级菜单
editmenu.add_command(label='我的收藏', command=bt15_callback)

editmenu = tk.Menu(menubar, tearoff=0)
# 一级菜单
menubar.add_cascade(label='照片滤镜', menu=editmenu)
# 二级菜单
submenu3 = tk.Menu(submenu1)
editmenu.add_cascade(label='简笔画', menu=submenu3)
submenu3.add_command(label='选择图片', command=bt21_callback)
submenu3.add_command(label='上妆图片', command=bt19_callback)
submenu3.add_command(label='素颜图片', command=bt20_callback)
editmenu.add_command(label='美白磨皮', command=skinSmoothing_callback)

editmenu = tk.Menu(menubar, tearoff=0)


root.config(menu=menubar)

backgroundImg=resize_pic2(current_path + "\\imgs\\background2.jpeg")
canvas=tk.Canvas(root,width=670,height=560)
canvas.create_image(335,280,image=backgroundImg)
canvas.pack()

frame1 = tk.Frame(root)
before_img = tk.PhotoImage(file=current_path + "\\imgs\\before_makeup_default.png")
label1 = tk.Label(frame1, image=before_img, relief="raised", bd=4)
label1.pack(side="bottom")
frame1.place(x=50, y=10)

frame2 = tk.Frame(root)  # root主窗口的第二个框架
after_img = tk.PhotoImage(file=current_path + "\\imgs\\after_makeup_default.png")
label2 = tk.Label(frame2, image=after_img, relief="raised", bd=4)
label2.pack(side="bottom")
frame2.place(x=350, y=10)

frame3 = tk.Frame(root)
img = []
for i in range(1, 10):
    img.append(resize_pic1(current_path + "\\imgs\\daily1\\" + str(i) + ".jpg"))

var = tk.IntVar()
r = []
for i in range(9):
    r.append(tk.Radiobutton(frame3, variable=var, value=i + 1, command=get_daily1_path,
                            indicatoron=False, width=65, height=110, image=img[i]))
    r[i].pack(side="left")

frame3.place(x=10, y=300)

# Default before-makeup path and after-makeup path
global before_makeup_path
global after_makeup_path
before_makeup_path = 'imgs/no_makeup/xfsy_0055.png'
after_makeup_path = 'imgs/makeup/XMY-136.png'

# block4，设置上妆的button
img_sz=Image.open(current_path + "\\imgs\\shangzhuang.png")
img_sz = img_sz.resize((165, 60), Image.ANTIALIAS)
img_sz = ImageTk.PhotoImage(image=img_sz)
bt4 = tk.Button(relief=FLAT,image=img_sz,command=bt4_callback)#上妆功能
bt4.place(x=270,y=450)


# block11，设置收藏的button
star=resize_pic3(current_path + "\\imgs\\shoucang1.jpeg")
bt11 = tk.Button(relief=FLAT,image=star, command=bt11_callback)#收藏功能
bt11.place(x=440,y=250)

root.mainloop()  # 主窗口循环显示
