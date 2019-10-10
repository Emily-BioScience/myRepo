# -*- coding: UTF-8 -*-
# UTF-8的注释，是为了兼容中文

import turtle as t

# 方形和圆形
def square_circle():
    # start
    t.penup()
    t.goto(100, 100)  # to straight to the point
    t.pendown()

    # square
    t.goto(100, -100)
    t.goto(-100, -100)
    t.goto(-100, 100)
    t.fd(200)   # go front
    # t.bk(100)   # go back

    # circles
    for head in [0, 90, 180, 270]:
        t.seth(head)  # change angle
        t.circle(100, 360)  # draw a circle with r=100 and angle=360


# 八边形
def octagon():
    # start
    t.penup()
    t.goto(400, 200)
    t.pendown()

    # octagon
    for i in range(8):
        t.fd(100)
        t.left(45)


# 八边形，小版
def octagon2():
    # start
    t.penup()
    t.goto(300, 0)
    t.pendown()

    # octagon2
    for i in range(8):
        t.fd(150)
        t.left(135)


# 蛇
def snake():
    t.fd(100)
    t.left(90)
    t.fd(100)
    t.left(45)
    t.fd(140)



if __name__ == "__main__":
    # setup
    t.setup()
    t.colormode(1.0)  # 1.0, 255
    t.pencolor('yellow')  # t.pencolor(0.63, 0.13, 0.94); t.pencolor((0.63, 0.13, 0.94))
    t.pensize(5)

    # draw
    square_circle()
    octagon()
    octagon2()
    # snake()

    # exit
    t.hideturtle()
    t.exitonclick()



# 画笔控制函数
# t.penup()
# t.pendown()
# t.pensize()
# t.pencolor()

# 运动控制函数
# t.forward(d)
# t.circle(r, extent)  # 圆心为左侧距离为r的地方

# 方向控制函数
# t.seth(angel)  # setheading，控制海龟面对的方向
# t.left(angel)  # 在左转方向上旋转的角度
# t.right(angel)  # 在右转方向上旋转的角度