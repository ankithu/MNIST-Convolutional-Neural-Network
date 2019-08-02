from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import Image
import cv2 
import tensorflow as tf 
import time

CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
model = tf.keras.models.load_model("mnist-cnn-dense")
class Paint(object):
    
    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'white'

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.color_button = Button(self.root, text='color', command=self.choose_color)
        self.color_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.predict_button = Button(self.root, text='predict', command=self.predict)
        self.predict_button.grid(row=0, column=5)

        self.choose_size_button = Scale(self.root, from_=20, to=100, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg='white', width=280, height=280)
        self.c.grid(row=1, columnspan=5)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.c.create_rectangle(0,0,280,280,fill='black')

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'black' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def predict(self):
        print("making prediction")
        self.c.postscript(file="digit.eps") # save canvas as encapsulated postscript
        img = Image.open("digit.eps")
        img.save("digit.png", "png")
        time.sleep(0.2)
        prediction = model.predict([self.prepare()])
        print(prediction)
        count = 0
        for num in prediction[0]:
            if num == 1:
                print(count)
            count += 1

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def prepare(self):
        IMG_SIZE = 28
        img_array = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


if __name__ == '__main__':
    Paint()