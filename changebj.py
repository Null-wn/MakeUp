#更换背景
from PIL import Image
import os


def changeback(path1, path2):
    # 读取底片

    imp = Image.open(path2)
    print(imp.width, imp.height)
    imp = imp.resize(size=(600, 600))


    # 读取要粘贴的图片 RGBA模式

    imq = Image.open(path1)
    print(imq.width, imq.height)
    imq = imq.resize(size=(600, 600))
    # 分离通道

    r, g, b, a = imq.split()

    # 粘贴

    imp.paste(imq, (0, 0, imq.width, imq.height), mask=a)
    imp.show()
    global i
    i=i+1
    imp = imp.resize(size=(500, 500))
    imp.save('D:/dachuang/makeup/imgs/changebj/'+str(i)+'.jpg')

def main():
    path1 = 'humanseg_output/7.png'  # 文件目录
    path0='bj/'
    global i
    i=0
    paths = [path0 + i for i in os.listdir(path0)]  # 获取背景图列表
    for path in paths:
        changeback(path1, path)


if __name__ == '__main__':
    main()
