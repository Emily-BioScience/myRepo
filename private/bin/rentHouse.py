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


if __name__ == '__main__':
    # data = {'days': [10, 10, 10, 10, 30, 30, 30, 30, 5, 3, 2],
    #         'price': [8800, 8700, 8400, 8300, 8600, 8600, 8900, 9300, 8500, 8500, 8500],
    #         'discount': [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1]}
    data = {'days': [2, 3, 4, 7, 10, 20, 30],
            'price': [8500, 8500, 8500, 8600, 8700, 8900, 9200],
            'discount': [1, 1, 1, 1, 1, 1, 1]}
    data = pd.DataFrame(data)
    for i in data.index:
        line = data.ix[i]
        get = real_get(line['price'], line['days'], line['discount'])
        print("免租期：{}天，包物业取暖：{}\n公司报价：{} => 实得{:.2f}\n\n".format(line['days'], line['discount'], line['price'], get))
    # with_call = real_get(8800, 10, True)
    # with_back = real_get(8700, 10, True)
    # print("免租期10天，我们不管物业取暖：\n谈判价：8800 => 实得{:.2f}\n底价：8700 => 实得{:.2f}\n".format(with_call, with_back))
    # without_call = real_get(8400, 10, False)
    # without_back = real_get(8300, 10, False)
    # print("免租期10天，我们包物业取暖：\n谈判价：8400 => 实得{:.2f}\n底价：8300 => 实得{:.2f}\n".format(without_call, without_back))

