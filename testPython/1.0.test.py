# -*- encoding: UTF-8 -*-
from lib import testModule as tm

if __name__ == '__main__':
    tm.mytest()


# with open('data/0.data.csv') as f:
#     data = {}
#     row = 0
#     for line in f:
#         item = line.strip().split(",")
#         row += 1
#         data[row] = item[::-1]
#
#     key = list(data.keys())
#     key.sort(reverse=True)
#
#     for k in key:
#         str = ";".join(data[k])
#         str = ''.join(str.split())
#         print(str)


    # items = list(key.items())
    # items.sort(key=lambda x:x[1], reverse=True)  # 按总出现频率从大往小排
    # print("No. of keys:", len(items))
    # print("No. of lines:", count)



# with open('data/0.latex.log', encoding='utf-8') as f:
#     row = 0
#     col = 0
#     for line in f:
#         line_cols = len(line) - 1
#         if line_cols:  # 非空行
#             row += 1
#             col += line_cols
#     print("{:d}".format(round(col/row)))

# s = '''双儿 洪七公 赵敏 赵敏 逍遥子 鳌拜 殷天正 金轮法王 乔峰 杨过 洪七公 郭靖
#        杨逍 鳌拜 殷天正 段誉 杨逍 慕容复 阿紫 慕容复 郭芙 乔峰 令狐冲 郭芙
#        金轮法王 小龙女 杨过 慕容复 梅超风 李莫愁 洪七公 张无忌 梅超风 杨逍
#        鳌拜 岳不群 黄药师 黄蓉 段誉 金轮法王 忽必烈 忽必烈 张三丰 乔峰 乔峰
#        阿紫 乔峰 金轮法王 袁冠南 张无忌 郭襄 黄蓉 李莫愁 赵敏 赵敏 郭芙 张三丰
#        乔峰 赵敏 梅超风 双儿 鳌拜 陈家洛 袁冠南 郭芙 郭芙 杨逍 赵敏 金轮法王
#        忽必烈 慕容复 张三丰 赵敏 杨逍 令狐冲 黄药师 袁冠南 杨逍 完颜洪烈 殷天正
#        李莫愁 阿紫 逍遥子 乔峰 逍遥子 完颜洪烈 郭芙 杨逍 张无忌 杨过 慕容复
#        逍遥子 虚竹 双儿 乔峰 郭芙 黄蓉 李莫愁 陈家洛 杨过 忽必烈 鳌拜 王语嫣
#        洪七公 韦小宝 阿朱 梅超风 段誉 岳灵珊 完颜洪烈 乔峰 段誉 杨过 杨过 慕容复
#        黄蓉 杨过 阿紫 杨逍 张三丰 张三丰 赵敏 张三丰 杨逍 黄蓉 金轮法王 郭襄
#        张三丰 令狐冲 赵敏 郭芙 韦小宝 黄药师 阿紫 韦小宝 金轮法王 杨逍 令狐冲 阿紫
#        洪七公 袁冠南 双儿 郭靖 鳌拜 谢逊 阿紫 郭襄 梅超风 张无忌 段誉 忽必烈
#        完颜洪烈 双儿 逍遥子 谢逊 完颜洪烈 殷天正 金轮法王 张三丰 双儿 郭襄 阿朱
#        郭襄 双儿 李莫愁 郭襄 忽必烈 金轮法王 张无忌 鳌拜 忽必烈 郭襄 令狐冲
#        谢逊 梅超风 殷天正 段誉 袁冠南 张三丰 王语嫣 阿紫 谢逊 杨过 郭靖 黄蓉
#        双儿 灭绝师太 段誉 张无忌 陈家洛 黄蓉 鳌拜 黄药师 逍遥子 忽必烈 赵敏
#        逍遥子 完颜洪烈 金轮法王 双儿 鳌拜 洪七公 郭芙 郭襄 赵敏'''
#
# s.replace("\n", "")
# words = s.split(" ")  # 确定的分隔符
# # words = jieba.lcut(s)  # 效果不算太好
#
# count = {}
# for w in words:
#     if len(w) == 1:  # 空格字符
#         # print(">>>> #{}#".format(w))
#         pass
#     elif w:  # 去除空字符
#         count[w] = count.get(w, 0) + 1
#
#
# items = list(count.items())
# items.sort(key=lambda x:x[1], reverse=True)
# person, frequency = items[0]
# print(person)

# for person, frequency in items:
#     print("person={}; frequency={}".format(person, frequency))