from tkinter import *
import os
from PIL import ImageTk,Image

class Display(object):

    def __init__(self):
        self.images = [];
        self.imgIndex = 0;
        self.master= Tk()
        self.framePhoto = Frame(self.master, bg='gray50',relief = RAISED, width=800, height=600, bd=4)
        self.framePhoto.pack();
        prevBtn = Button(self.framePhoto, text='Previous', command=lambda s=self: s.getImgOpen('prev')).place(relx=0.85, rely=0.99, anchor=SE)
        nextBtn = Button(self.framePhoto, text='Next', command=lambda s=self: s.getImgOpen('next')).place(relx=0.90, rely=0.99, anchor=SE)
        #prevBtn.pack();
        #nextBtn.pack();
        self.getImgList('test_2/test_2','.bmp')
        mainloop()

    def getImgList(self, path, ext):
        imgList = [os.path.normcase(f) for f in os.listdir(path)]
        imgList = [os.path.join(path, f) for f in imgList if os.path.splitext(f)[1] == ext]
        self.images.extend(imgList)
        #print self.images

    def getImgOpen(self,seq):
        print ('Opening %s' % seq)
        if seq=='ZERO':
            self.imgIndex = 0
        elif (seq == 'prev'):
            if (self.imgIndex == 0):
                self.imgIndex = len(self.images)-1
            else:
                self.imgIndex -= 1
        elif(seq == 'next'):
            if(self.imgIndex == len(self.images)-1):
                self.imgIndex = 0
            else:
                self.imgIndex += 1

        self.masterImg = Image.open(self.images[self.imgIndex])
        self.master.title(self.images[self.imgIndex])
        self.masterImg.thumbnail((400,400))
        self.img = ImageTk.PhotoImage(self.masterImg)
        label = Label(image=self.img)
        label.image = self.img # keep a reference!
        label.pack()
        label.place(x=100,y=100)
        #self.lbl['image'] = self.img
        return

d = Display();
d.getImgOpen('next')
