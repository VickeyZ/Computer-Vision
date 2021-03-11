import cv2
import numpy as np
import os
import sys

def file_dir(root):
    image = list()
    video = ""
    for file in os.listdir(root):
        if file.endswith(".jpg"):
            image.append(os.path.join(root, file))
        elif file.endswith(".avi"):
            video = os.path.join(root, file)
    return image,video

def video_write(img):
    cv2.putText(img, '3180103770 ZhangWenqi', (500, 500), cv2.FONT_HERSHEY_COMPLEX, 1.0, (200, 200, 100), 5)
    videoWriter.write(img)

class Animation:  # 片头，方块碰撞动画制作
    def __init__(self,size:"图像尺寸元组",r:"方块中心到顶点的距离",start:"起始位置元组",step:"每一帧移动几个像素"):
        self.sizex = size[0]
        self.sizey = size[1]
        self.step = step
        self.r = r
        
        self.vx = 1   # 初始速度
        self.vy = 1
        self.x = start[0]
        self.y = start[1]
        
        assert not self.collidex()
        assert not self.collidey()
    def collidex(self):return False if self.r-1 < self.x < self.sizex-self.r else True
    def collidey(self):return False if self.r-1 < self.y < self.sizey-self.r else True
    def __iter__(self):
        while True:
            canvas = np.ones((self.sizey,self.sizex,3),dtype='uint8')*255
            x,y,r = self.x, self.y, self.r
            for dx in range(r):
                for dy in range(r-dx):
                    canvas[y+dy][x+dx] = (0,0,0)
                    canvas[y+dy][x-dx] = (0,0,0)
                    canvas[y-dy][x-dx] = (0,0,0)
                    canvas[y-dy][x+dx] = (0,0,0)
            yield canvas
            for _ in range(self.step):
                if self.collidex():self.vx *= -1    # 若碰撞则反向
                if self.collidey():self.vy *= -1
                self.x += self.vx
                self.y += self.vy

def translation(last_frame,current_frame):
    for index in range(1,61):
        new_frame=last_frame
        cut_height = int(size[1]*index/60)
        for i in range(cut_height-1,-1,-1):
            for j in range(0,size[0]):
                new_frame[size[1]+i-cut_height,j]=current_frame[i,j]
        video_write(new_frame)
    for i in range(40):
        video_write(current_frame)

if __name__ == "__main__":
    root = sys.argv[1]
    image, video = file_dir(root)

    #读取视频格式
    cap = cv2.VideoCapture(video) #1280,544,24
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(size,fps)

    videoWriter = cv2.VideoWriter(os.path.join(root,"output.avi"),cv2.VideoWriter_fourcc('M','J','P','G'),fps,size)
    print("-------LOADING-------")

    #片头动画制作
    for img,_ in zip(Animation(size=size,r=70,start=(200,125),step=5),range(300)):
        video_write(img)
        
    #图片处理，逐帧切换

    # for i in range(5 * fps):
    #     if i < 5 * fps:
    #         img = cv2.imread(os.path.join(root,'%d.jpg') % (i//fps + 1))
    #         img = cv2.resize(img, (1280,544))

    #         #add text on the figure
    #         cv2.putText(img, text, (500, 500), cv2.FONT_HERSHEY_COMPLEX, 1.0, (200, 200, 100), 5)
    #         videoWriter.write(img)
    #     else:
    #         break

    for i in range(len(image)):
        img = cv2.imread(image[i])
        img = cv2.resize(img, size)
        img_next = img.copy()
        if i%2==0:
            for j in range(50):
                video_write(img)
        else:
            translation(img_next, img)


    #片尾插入视频             
    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret:
            video_write(frame)
        else:
            break        
    videoWriter.release()  
    print("-------DOWN-------")