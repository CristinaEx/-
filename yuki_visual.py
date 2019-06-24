import sys
from tkinter import *
import datetime
import time
import os
from PIL import Image
from PIL import ImageTk

class YukiVisual(Frame):
    
    def __init__(self,parent = None):
        YukiVisual.received_text = str()
        Tk().title('ene~~~')
        Frame.__init__(self,parent)
        frame_left_top = Frame(self,width=380, height=370, bg='white')
        frame_left_center  = Frame(self,width=380, height=200, bg='white')
        frame_button = Frame(self,width=380, height=30)
        frame_left_bottom  = Frame(frame_button,width=350, height=30)
        frame_right_bottom  = Frame(frame_button,width=30, height=30)        
        frame_right = Frame(self,width=426, height=600, bg='white')
        ##创建需要的几个元素
        self.__text_msglist = Text(frame_left_top)
        self.__text_msg = Text(frame_left_center)
        Button(frame_left_bottom, text='submit', command=self.button_click_first).grid(row=0, column=1) 
        Button(frame_left_bottom,text='clear', command=self.button_click_second).grid(row=0, column=2) 
        Button(frame_right_bottom,text='ESC', command=self.button_click_third).grid(row=0, column=3) 
        #创建一个绿色的tag
        self.__text_msglist.tag_config('green', foreground='#008B00')
        #pic 426*600
        pic_root = Image.open('pic_1.gif')
        pic_root = ImageTk.PhotoImage(pic_root)
        pic_label = Label(frame_right, image = pic_root)
        pic_label.bm = pic_root
        pic_label.pack()
        #使用grid设置各个容器位置
        frame_left_top.grid(row=0, column=0, padx=2, pady=5)
        frame_left_center.grid(row=1, column=0, padx=2, pady=5)
        frame_left_bottom.grid(row=0, column=0)
        frame_right_bottom.grid(row=0, column=1)
        frame_button.grid(row=2, column=0,rowspan=2)
        frame_right.grid(row=0, column=1, rowspan=3, padx=4, pady=5)
        frame_left_top.grid_propagate(0)
        frame_left_center.grid_propagate(0)
        frame_left_bottom.grid_propagate(0)
        #把元素填充进frame
        self.__text_msglist.grid()
        self.__text_msg.grid()
        self.pack()
        self = mainloop()

    def button_click_first(self):
        self.received_text = self.__text_msg.get('0.0', END)
        msgcontent = 'master : ' + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + '\n ' 
        self.__text_msglist.insert(END, msgcontent, 'green')
        self.__text_msglist.insert(END, self.received_text)
        self.__text_msg.delete('0.0', END)
        self.send_message('')

    def button_click_second(self):
        self.__text_msg.delete('0.0', END)

    def button_click_third(self):
        exit(0)

    def send_message(self,text):
        msgcontent = 'ene : ' + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + '\n '
        self.__text_msglist.insert(END, msgcontent, 'green')
        self.__text_msglist.insert(END, text)


class frame_talk(Frame):
    def __init__(self,parent = None):
        Frame.__init__(self,parent)

if __name__ == '__main__':    
    vis = YukiVisual()
    
    
