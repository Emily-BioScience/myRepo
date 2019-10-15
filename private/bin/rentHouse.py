# -*- coding UTF-8 -*-
import pandas as pd


def real_get(price, days, discount):
    if discount:
        discount = 2753.10 + 2022.99
    else:
        discount = 0
    get = price * 12 / 365 * (365 - days) - discount
    get = get/12
    return(get)


def calculate_price():
    data = {'days': [2, 3, 4, 7, 10, 20, 30],
            'price': [8500, 8500, 8500, 8600, 8700, 8900, 9200],
            'discount': [1, 1, 1, 1, 1, 1, 1]}
    data = pd.DataFrame(data)
    for i in data.index:
        line = data.ix[i]
        get = real_get(line['price'], line['days'], line['discount'])
        print("免租期：{}天，包物业取暖：{}\n公司报价：{} => 实得{:.2f}\n\n".format(line['days'], line['discount'], line['price'], get))


def print_pay_details(date, time, price):
    year, month, day = date
    for i in range(len(price)):
        new_month, new_year = month + time[i], year
        if new_month / 12 > 2:  # 两年后
            new_month, new_year = new_month - 24, new_year + 2
        elif new_month / 12 > 1:  # 一年后
            new_month, new_year = new_month - 12, new_year + 1
        print("\t第{}次：{}-{}-{},交{}个月租金".format(i+1, new_year, new_month, day, price[i]))


def pay_three_first(date):
    time = [0, 3, 6, 9, 12, 15, 18, 21]
    price = [3, 3, 3, 3, 3, 3, 3, 2]
    print("\n>>>方式：首次付3个月，最后一期付2个月")
    print_pay_details(date, time, price)


def pay_two_first(date):
    time = [0, 2, 5, 8, 11, 14, 17, 20]
    price = [2, 3, 3, 3, 3, 3, 3, 3]
    print("\n>>>方式：首次付2个月，其他期提前1个月付")
    print_pay_details(date, time, price)


if __name__ == '__main__':
    # 计算底价
    # calculate_price()

    # 计算支付方式
    date = [2019, 10, 15]
    pay_three_first(date)
    pay_two_first(date)
