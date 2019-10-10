# -*- coding: UTF-8 -*-
# UTF-8的注释，是为了兼容中文

# temperature convertion
def tempConvert(temp):
    if temp[-1] in ['F', 'f']:
        new = (eval(temp[0:-1])-32)/1.8
        return("New: {:.2f} C".format(new))
    elif temp[-1] in ['C', 'c']:
        new = eval(temp[0:-1])*1.8+32
        return("New: {:.2f} F".format(new))
    else:
        return('error')



if __name__ == "__main__":
    while True:
        temp = input("Please input a number: ")
        convert = tempConvert(temp)
        if convert == 'error':
            print("error")
            break
        else:
            print(convert)