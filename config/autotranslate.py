# -*- coding UTF-8 -*-
import re
import execjs
import urllib,urllib2
import sys
import json
import os.path
from ruamel import yaml
from PyPDF2.pdf import PdfFileReader, PdfFileWriter, ContentStream

reload(sys)
sys.setdefaultencoding( "utf-8" )


def getFiles(file):
    files = []
    with open(file, encoding = 'utf-8') as f:
        content = yaml.load(f, Loader=yaml.RoundTripLoader)
        for f in content.keys():
            files.append(f)
    return(files)


def autotranslate(infile, outfile):
    content = readPDFfile(infile)
    # results = translateContent(content)


def readPDFfile(infile):
    pdf = PdfFileReader(infile, "rb"))
    content = ""
    num = pdf.getNumPages()
    for i in range(0, num):
        extractedText = pdf.getPage(i).extractText()
        content +=  extractedText + "\n"
    return content


def dopage(page):
    content = page["/Contents"].getObject()
    if not isinstance(content, ContentStream):
        content = ContentStream(content, pdf)

    text = u_("")
    for operands, operator in content.operations:
        # print operator, operands
        if operator == b_("Tj"):
            _text = operands[0]
            if isinstance(_text, TextStringObject):
                text += _text + " "
        elif operator == b_("rg"):
            text += "\n"
        elif operator == b_("T*"):
            text += "\n"
        elif operator == b_("'"):
            text += "\n"
            _text = operands[0]
            if isinstance(_text, TextStringObject):
                text += operands[0] + " "
        elif operator == b_('"'):
            _text = operands[2]
            if isinstance(_text, TextStringObject):
                text += _text + " "
        elif operator == b_("TJ"):
            for i in operands[0]:
                if isinstance(i, TextStringObject):
                    text += i
            text += " "

    texts = text.split('. ')
    results = ''
    for i in range(len(texts)):
        try:
            results = results + translate(str(texts[i])) + "\n"
        except Exception as e:
            print
            e
    return results



if __name__ == '__main__':
    files = getFiles('conf/autotranslate.yaml')
    for infile in files:
        outfile = os.path.basename(infile).replace('.pdf', '.out.pdf')
        autotranslate(infile, outfile)
