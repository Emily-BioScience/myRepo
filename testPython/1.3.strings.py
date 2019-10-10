# -*- coding: UTF-8 -*-
# UTF-8的注释，是为了兼容中文

# 字符串的遍历
def string_enumerate(mystr):
    print("\nEnumerate:\nmystr=", mystr)
    for i in range(len(mystr)):  # 从0开始，到len-1结束，遍历整个字符串
        print("i=", i, ", string[i]=", mystr[i])


# 字符串的索引
def string_slice(mystr):
    print("\nSlice:\nmystr=", mystr)
    print("mystr[1:5]=", mystr[1:5])  # 字符串切片
    print("mystr[1:5:3]=", mystr[1:5:3])  # 字符串切片，步长为3
    print("mystr[::-1]=", mystr[::-1])  # 将字条串进行逆序，-1表示从后从前切片


# 字符串的转义符，涉及单引号、双引号、不可打印的含义
def string_transform(mystr):
    print("\nTransform:\nmystr=", mystr)
    print("\" => ","\"",mystr)   # 转双引号
    print("\' => ","\'",mystr)   # 转单引号
    print("\\b => ","\b",mystr)  # 回退，没太明白？？？
    print("\\n => ","\n",mystr)  # 换行，光标移动至下行首
    print("\\r => ","\r",mystr)  # 回车，光标移动至本行首，没太明白？？？


# 字符串操作符
def string_manipulate(mystr, mystr2):
    print("\nManipulate:\nmystr=", mystr, ", mystr2=", mystr2)
    newstr = mystr + mystr2  # 连接两个字符串
    print(mystr, "+", mystr2, '=', newstr)
    multistr = mystr * 3   # 复制一个字符串，整数次
    print(mystr, "* 3 =", multistr)
    if (mystr in mystr2):  # 判断mystr是否mystr2的子串
        print(mystr, "is in", mystr2)
    else:
        print(mystr, "is not in", mystr2)


# 字符串处理函数
def string_function():
    mystr = 'testhaha'
    mynum = 425
    mylist = [1, 2]

    print("\nFunction:\nmystr =", mystr)
    print("len(mystr) =", len(mystr))
    print("str(mynum) =", str(mylist))   # 加双引号，变成字符串，跟eval（去掉双引号）是一对
    print("type(str(mynum)) =", type(mylist))   # 加双引号，变成字符串，跟eval（去掉双引号）是一对
    print("oct(mynum) =", oct(mynum))  # 将二进制转换成八进制
    print("hex(mynum) =", hex(mynum))  # 将二进制转换成十六进制
    print("chr(116) =", chr(10004))   # 将Unicode编码，转换成对应的字符
    print("ord(t) =", ord('t'))   # 将字符，转换成对应的Unicode编码

    for i in range(12):
        print(str(chr(9800+i)), end="")  # 12个星座，end选项，可以让输出不换行


# 字符串处理方法，字符串和变量本身也是对象，所以存在一些操作方法
def string_method():
    mystr = "Hello World"
    print("\nMethod:\nmystr =", mystr)
    print("mystr.upper() =", mystr.upper())
    print("mystr.lower() =", mystr.lower())
    print("mystr.split() =", mystr.split())  # 按分隔符分割字符串，获得子串列表，默认分隔符为空格
    print("mystr.count(sub) =", mystr.count('l'))  # 数数，看子串出现的次数
    print("mystr.replace(old, new) =", mystr.replace('World', 'New World'))  # 替换子串
    print("mystr.center(20, '=') =", mystr.center(30, '='))  # 将字符串居中，两端加===，总长为20，打印规则处理
    print("mystr.strip(Hed) =", mystr.strip('Held'))  # 将左右两侧出现的H、e、d字符删除
    print("mystr.join(string) =", mystr.join('12345'))  # 把join里的字符串拆成单个字符，中间插入mystr


# 字符串的格式化输出：字符串.format()方法来格式化， {}叫槽，为点位符，所有标点符号用英文输入
def string_format():
    mynum = 425
    mystr = 'World'
    print("\nFormat:\nmynum =", mynum)
    print("{}:test, {}:haha".format(mynum, mystr))  # 左侧，填充、对齐、宽度，第一定宽度，第二定对齐试，第三定填充字符
    print("{:*>20}:test, {}:haha".format(mynum, mystr))    # :引号符号， * 填充，右对齐，宽度为20
    print("{:*^20}:test, {}:haha".format(mynum, mystr))    # :引号符号， * 填充，中对齐，宽度为20
    print("{:*<20}:test, {}:haha".format(mynum, mystr))    # :引号符号， * 填充，左对齐，宽度为20
    print("{:->20}:test, {}:haha".format(mynum, mystr))    # :引号符号， - 填充
    print("{:-<20,}:test, {}:haha".format(mynum, mystr))   # :引号符号， ','表示千为单位分隔
    print("{:-^20,.2f}:test, {}:haha".format(mynum, mystr))  # :引号符号， '.2f'表示只输出浮点数的两位小数点
    print("{:b}:test, {:s}:haha".format(mynum, mystr))  # :引号符号，':b'表示输出二进制数字
    print("{:o}:test, {:s}:haha".format(mynum, mystr))  # :引号符号，':o'表示输出八进制数字
    print("{:d}:test, {:s}:haha".format(mynum, mystr))  # :引号符号，':d'表示输出十进制数字    #################
    print("{:x}:test, {:s}:haha".format(mynum, mystr))  # :引号符号，':x'表示输出十六进制进制数字
    print("{:e}:test, {:s}:haha".format(mynum, mystr))  # :引号符号，':e'表示输出科学记数法数字
    print("{:f}:test, {:s}:haha".format(mynum, mystr))  # :引号符号，':f'表示输出浮点数数字    #################
    print("{:c}:test, {:s}:haha".format(mynum, mystr))  # :引号符号，':c'表示字符形式，即Unicode编码形式


# 星期几的转换程序
def weekday_transform():
    while 1:
        weeks = "一二三四五六日"
        num = eval(input("Input a number for weekday transforming, [1-7]: "))
        if num>=1 and num<=7:
            weekday = "星期" + weeks[num-1]
            print(num, "=> ", weekday)
        else:
            print("错误：只能输入1-7的数字")
            break



if __name__ == '__main__':
    # mystr = 'Hello world'
    # mystr2 = 'Hello\' \'world'
    # string_enumerate('Hello world')
    # string_enumerate('Hello\' \'world')
    # string_slice('Hello world')
    # string_transform('Hello\' \'world')
    # string_manipulate('Hello', 'Hello world')

    # weekday_transform()
    # string_function()
    # string_method()
    string_format()





# # 字符串的两类，4种表示方式
# str1 = 'test'
# str2 = "test"
# str3 = '''te
# st'''
# str4 = """t
# es
# t"""
# print("str1:", str1)
# print("\nstr2:", str2)
# print("\nstr3:", str3)
# print("\nstr4:", str4)
#
#
# # 如果字符串中含有一到两种字符串
# str5 = 'test"haha"' # 外面单引号，内部双引号被视为字符
# str6 = "test'haha'"  # 外面双引号，内部单引号被视为字符
# str7 = '''test'haha'"haha"'''  # 外面三单引号，内部各引号都被视为字符
# print("str5:", str5)
# print("\nstr6:", str6)
# print("\nstr7:", str7)



