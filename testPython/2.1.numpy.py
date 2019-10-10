# -*- coding UTF-8 -*-
import numpy as np
import numpy.random as npr
from PIL import Image


def usenumpy(infile, outfile):
    a = np.array([[0.0, 1, 2], [3, 4, 5]],)  # 从python的列表、元组、列表和元组混合类型等多种类型，生成一个ndarray的数组
    b = np.array([[9.0, 8, 7], [6, 5, 4]], dtype='float64')
    print(">>>>> 基本属性\nndim: ", a.ndim)  # 秩，rank，轴的数量
    print("shape: ", a.shape)  # 尺度，对于矩阵，n行m列，# 轴，axis，数据的维度
    print("size: ", a.size)  # 元素的个数，相当于n*m
    print("dtype: ", a.dtype)  # 对象的元素类型, int32(intc), int64(intc), bool, intp(用于索引的整数)，int8（字节长度的整数），unit32(32位无符号整数)，float浮点数, complex128(复数：实部和虚部64位浮点数)
    print("itemsize: ", a.itemsize)  # 每个元素的大小，以字节为单位

    c = a + b
    print("\n>>>>> 数组运算\na: ", a)
    print("b: ", b)
    print("a + b: ", c)

    print("\n>>>>> 数组创建\narange(10): ", np.arange(10))  # 返回ndarray类型，元素从0到n-1
    print("ones(4, 2): ", np.ones((4, 2)))  # 返回一个全1数组，shape是元组类型
    print("zeros(4, 2): ", np.zeros((4, 2)))  # 返回一个全0数组，shape是元组类型
    print("full((4, 2), 3): ", np.full((4, 2), 3))  # 返回一个维度为shape的数组，每个元素值都是val
    print("eye(4): ", np.eye(4))  # 返回一个单位矩阵，n*n, 对角线为1，其余为0

    print("\n>>>>> 数组创建2\nones_like(a): ", np.ones_like(a))  # 返回一个与a相同维度的数组，元素为1
    print("zeros_like(a): ", np.zeros_like(b))  # 返回一个与b相同维度的数组，元素为0
    print("full_like(a, 8): ", np.full_like(a, 8))  # 返回一个与a相同维度的数组，元素为val


    print("\n>>>>> 数组创建3\nnp.linspace(1, 100, 4): ", np.linspace(1, 100, 4, endpoint=True))  # 生成浮点数组，根据起止数据等间距的填充数据，等差数列，endpoint设置结束数值是否出现
    print("np.concatenate((a, b)): ", np.concatenate((a, b)))  # 将两个或多个数组合并成一个数组

    print("\n>>>>> 数组修改\nnp.reshape((1, 6)): ", a.reshape((1, 6)))  # 不改变数组元素，返回一个shape形状的数组，原数组不变
    print("a: ", a)
    print("a.resize((1, 6)): ", a.resize((1, 6)))  # 与reshape功能一致，但修改原数组
    print("a: ", a)
    print("b.swapaxes(): ", b.swapaxes(0, 1))   # 将数组n个维度中两个维度进行调换
    print("b.flatten(): ", b.flatten())  # 对数组进行降维，返回折叠后的一维数组，原数组不变
    print("b: ", b)

    new_b = b.astype("int64")  # 类型变换，一定会创建新的数组
    print("\n>>>>> 数组修改2\nnp.astype(int64): ", new_b)
    print("b: ", b)
    print("b.tolist(): ", b.tolist())

    a = np.array([1, 2, 3, 4, 5])
    print("\n>>>>> 数组操作\na[1:4:2]", a[1:4:2])  # 一维数组的索引和切片，0开始向右递增，-1开始向左递减
    a = np.arange(24).reshape((2, 3, 4))
    print("a: ", a)
    print("a.shape: ", a.shape)
    print("a[1, 2, 3]: ", a[1, 2, 3])  # 多维数组的索引和切片，先竖再横
    print("a[0:2, 2, 2:4]: ", a[0:2, 2, 2:4])  # 多维数组的索引和切片，先竖再横
    print("a[:, :, ::2]: ", a[:, :, ::2])  # 多维数组的索引和切片，先竖再横

    print("\n>>>>> 数组与标量运算\na/a.mean(): ", a/a.mean())  # 标量运算，各个元素自己算
    print("npr.rand(3, 2, 2): ", npr.rand(3, 2, 2))  # 生成[0, 1]之间的随机多维数组

    print("npr.randint(2, 4): ", npr.randint(2, 4))  # 生成[a, b]之间的随机整数
    a = 2
    b = 4
    print("random N-dimensional array [2, 4]: ", npr.rand(3, 2, 2) * (b-a) + a)  # 生成[a, b]之间的随机多维数组

    npr.seed(10)
    x = npr.rand(10, 2) - 1
    print("np.abs(x): ", np.abs(x))  # 元素级求绝对值
    print("np.fabs(x):", np.fabs(x))  # 元素级求绝对值
    print("np.sqrt(x):", np.sqrt(x))  # 元素级求平方
    print("np.square(x):", np.square(x))  # 元素级求平方根

    print("np.log(np.abs(x)):", np.log(np.abs(x)))  # 元素级求自然对数
    print("np.log2(np.abs(x)):", np.log2(np.abs(x)))  # 元素级求2的对数
    print("np.log10(np.abs(x)):", np.log10(np.abs(x)))  # 元素级求10的对数

    print("np.rint(x):", np.rint(x))  # 元素级求四舍五入值
    print("np.exp(x):", np.exp(x))  # 元素级求指数值
    print("np.sign(x):", np.sign(x))  # 元素级求符号，正，负，0

    y = np.random.rand(10, 2) - 1
    print("\n>>>>> 数组二元运算\nx-y ", x-y)  # 矩阵二元运算，各个元素自己算
    print("\nnp.mod(x, y) ", np.mod(x, y))  # 矩阵求模运算，即取余数
    print("x%y == np.mod(x, y)", x % y == np.mod(x, y))  # 布尔操作，各个元素各自比较

    data = np.loadtxt(infile, dtype=np.float, delimiter=',', unpack=False)
    print("data", data)
    np.savetxt(outfile, data, fmt='%.2f', delimiter=',')

    data2 = np.fromfile(infile, dtype=np.float, sep=',').reshape(1, 2)
    print("data2: ", data2)
    outfile2 = outfile.replace('.csv', '.txt')
    data2.tofile(outfile2, sep=",", format="%s")

    # 便捷存储，numpy格式文件
    outfile3  = outfile.replace('.csv', '.npy')
    np.save(outfile3, data)
    outfile4 = outfile.replace('.csv', '.npz')
    np.save(outfile4, data)

    npr.seed(10)
    print('x: ', x)
    print("shuffle(x): ", npr.shuffle(x))  # 按第1轴随机排列，改变数组a
    print("permutation(x): ", npr.permutation(x))  # 不改变数组a

    p = np.abs(x.flatten() / np.sum(x))
    print("npr.choice(1d-array, size, replace, p)", npr.choice(x.flatten(), (3, 2), p=p, replace=True))  # 从一维数组中抽取元素，形成(3, 2)的新数组，抽取概率为p，replace决定是否放回

    print("npr.uniform(low, high, size)", npr.uniform(1, 8, (3, 2)))  # 均匀分布
    print("npr.normal(mean, sd, size)", npr.normal(0, 1, (3, 2)))  # 正态分布
    print("npr.possion(p, size)", npr.poisson(0.5, (3, 2)))  # 泊松分布

    print("np.sum(x): ", np.sum(x))  # 总和，可以按轴求
    print("np.mean(x): ", np.mean(x))  # 平均值，可以按轴求
    print("np.average(x): ", np.average(x, 1, weights=x/np.sum(x)))  # 加权平均值，可以按轴求
    print("np.std(x): ", np.std(x))  # 标准差，可以按轴求
    print("np.var(x): ", np.var(x))  # 方差，可以按轴求

    print("np.max(x, 1): ", np.max(x, 1))  # 求最大值，可以按轴求
    print("np.min(x, 1): ", np.min(x, 1))  # 求最小值，可以按轴求
    print("np.argmin(x, 1): ", np.argmin(x, 1))  # 求最小值所在的一维坐标
    print("np.unravel_index(x.argmin(), x.shape): ", np.unravel_index(x.argmin(), x.shape))  # 将最小值所在的一维坐标，转换成数组的多维坐标
    print("np.ptp(x): ", np.ptp(x))  # 计算极差，最大值跟最小值之间的差值
    print("np.median(x): ", np.median(x))  # 计算中位数

    a, b = np.gradient(x)  # 连续值之间的变化率，即斜率，连续三个值，abc，b的梯度是(c-a)/2
    print("1st-dimension: np.gradient(x): ", a)   # 连续值之间的变化率，即斜率，在第一个维度上，存在两侧值，则后一个值减去前一个值，除以2，只有一侧值，当前值与旁边值的差
    print("2nd-dimension: np.gradient(x): ", b)   # 连续值之间的变化率，即斜率，在第二个维度上，存在两侧值，则后一个值减去前一个值，除以2，只有一侧值，当前值与旁边值的差


def usePIL(inImage, outImage):
    a = np.array(Image.open(inImage))
    print('image:\n', a.shape, a.dtype)
    b = [255, 255, 255] - a  # 反转
    im = Image.fromarray(b.astype('uint8'))
    im.save(outImage)

    a = np.array(Image.open(inImage).convert('L'))  # convert('L')，变成灰度图像
    print('image-grey:\n', a.shape, a.dtype)
    b = 255 * (a/255)**2 # 区间压缩，移动
    im = Image.fromarray(b.astype('uint8'))
    outImage2 = outImage.replace('.out', '.out2')
    im.save(outImage2)

    # 手绘风格
    depth = 10  # 取值范围，0-100
    grad_x, grad_y = np.gradient(a)
    grad_x = grad_x * depth / 100
    grad_y = grad_y * depth / 100

    vec_e1 = np.pi/2.2
    vec_az = np.pi/4.0
    dx = np.cos(vec_e1) * np.cos(vec_az)
    dy = np.cos(vec_e1) * np.sin(vec_az)
    dz = np.sin(vec_e1)

    A = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.0)
    uni_x = grad_x / A
    uni_y = grad_y / A
    uni_z = 1.0 / A
    b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)

    b = b.clip(0, 255)  # 裁剪至0-255 范围
    im = Image.fromarray(b.astype('uint8'))
    outImage3 = outImage.replace('.out', '.out3')
    im.save(outImage3)











if __name__ == '__main__':
    usenumpy('data/2.1.numpy.csv', 'output/2.1.numpy.out.csv')   # 强大的n维数组对象，ndarray，也可以叫数组
                                # 广播功能函数
                                # 整合C/C++/Fortran代码的工具
                                # 线性代数、傅里叶变换、随机数生成等功能
                                # 是SciPy, Pandas等数据处理或科学计算库的基础

    usePIL('./data/2.1.wuy.jpg', './output/2.1.wuy.out.jpg')

# ndarray
# 实际的数据
# 描述这些数据的元数据（数据维度、数据类型等）
# 一般要求所有元素类型相同（同质），下标从0开始，如果不同质，无法有效发挥numpy的优势，尽量避免使用

# 列表和数组的区别，列表中的元素，数据类型可以不同
# 一维数据：列表和集合类型
# 二维数据：列表类型
# 多维数据：列表类型
# 高维数据：字典类型或数据表示格式
# JSON, XML, YAML等格式，形成维度和关系