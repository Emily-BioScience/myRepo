# -*- coding: UTF-8 -*-
import os
from ruamel import yaml


def printVersion(version):
    if version == '0':
        version = ''
    else:
        version = "=={}".format(version)
    return(version)


def whetherInstall(current, desired):
    flag = 0
    if current == '0':
        flag = 1
    elif str(desired) > str(current):
        flag = 1
    return(flag)


def installLibs(file):
    # with open('library.log', 'w') as fo:
        #     yaml.dump(infor, fo, Dumper=yaml.RoundTripDumper)
    with open(file, encoding = 'utf-8') as f:
        content = yaml.load(f, Loader=yaml.RoundTripLoader)

        for pkg in content.keys():
            current = getCurrentVersion(pkg)   # current version
            desired = content[pkg].get('version', '0')  # assign version
            version = printVersion(desired)  # 根据输入，指定安装版本
            flag = whetherInstall(current, desired)

            print("omit {}\n".format(pkg))
            # 实际的自动安装
            if flag:
                print(">> {}{}\ncurrent: {}\ndesired: {}\nflag: {}\n".format(pkg, version, current, desired, flag))
                try:
                    print("# package {}\ncurrent version: {}".format(pkg, current))
                    os.system("/public/noncode/yangrui/miniconda3/bin/pip install {}{}".format(pkg, version))
                except:
                    print("failed to install package {}{}".format(pkg, version))
                else:
                    print("installed package {}{}\n".format(pkg, version))


def getCurrentVersion(pkg):
    current = '0'
    os.system("/public/noncode/yangrui/miniconda3/bin/pip list |grep {} >version.tmp".format(pkg))
    if os.path.exists('version.tmp'):
        with open('version.tmp') as f:
            for line in f:
                name, version = line.strip().split()
                if pkg == name and version > current:
                    current = version
                    # print(pkg, name, version, current)
        os.system("rm -f version.tmp")
    return(current)



if __name__ == '__main__':
    # 检查X11窗口的环境变量
    # print(os.environ['DISPLAY'])

    # 批量安装第三方库
    installLibs('./conf/autoinstall.yaml')




