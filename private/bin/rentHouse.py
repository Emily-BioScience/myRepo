# -*- coding UTF-8 -*-
import pandas as pd


def real_get(price, days, discount, year):
    if discount:
        discount = 2753.10 + 2022.99
    else:
        discount = 0
    get = price * (12 * year) / (365 * year) * (365 * year - days) - discount
    get = get/ (12 * year)
    return(get)


def calculate_price():
    global fo
    data = {'days': [10, 20, 30, 30, 30, 30, 30],
            'price': [8700, 8900, 9200, 9300, 9300, 9400, 9400],
            'discount': [1, 1, 1, 1, 1, 1, 1],
            'year': [1, 1, 1, 2, 1, 2, 1]}
    data = pd.DataFrame(data)
    for i in data.index:
        line = data.ix[i]
        get = real_get(line['price'], line['days'], line['discount'], line['year'])
        output = "租{}年，免租期{}天，包物业取暖：{}\n公司报价：{} => 实得{:.2f}\n\n".format(line['year'], line['days'], line['discount'], line['price'], get)
        fo.write(output)


def print_pay_details(date, early, time, price):
    global fo
    year, month, day = date
    if early:  # 提早多少天
        day = day - early
        if day <= 0:
            day, month = day + 30, month - 1
    tag = "\n>>>方式：首次付{}个月，其他期提前{}天付".format(time[0], early)
    fo.write(tag)
    for i in range(len(price)):  # 计算每一次支付的时间和数量
        new_month, new_year = month + time[i], year
        if new_month / 12 > 2:  # 两年后
            new_month, new_year = new_month - 24, new_year + 2
        elif new_month / 12 > 1:  # 一年后
            new_month, new_year = new_month - 12, new_year + 1
        if i:
            output = "\t第{}次：{}-{}-{},交{}个月租金".format(i+1, new_year, new_month, day, price[i])
        else:
            output = "\t第{}次：{}-{}-{},交{}个月租金".format(i+1, date[0], date[1], date[2], price[i])
        fo.write(output + "\n")


def pay_three_first(date):
    time = [0, 3, 6, 9, 12, 15, 18, 21]
    price = [3, 3, 3, 3, 3, 3, 3, 2]
    print_pay_details(date, 0, time, price)


def pay_two_first(date):
    time = [0, 3, 6, 9, 12, 15, 18, 21]
    price = [2, 3, 3, 3, 3, 3, 3, 3]
    print_pay_details(date, 30, time, price)
    print_pay_details(date, 20, time, price)
    print_pay_details(date, 15, time, price)
    print_pay_details(date, 10, time, price)
    print_pay_details(date, 5, time, price)
    print_pay_details(date, 0, time, price)



if __name__ == '__main__':
    # 输出到文件
    fo = open('../output/rentHouse.txt', 'w+')

    # 计算底价
    calculate_price()

    # 计算支付方式
    date = [2019, 10, 15]
    pay_three_first(date)
    pay_two_first(date)