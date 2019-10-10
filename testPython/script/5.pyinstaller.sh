#!/usr/bin/env bash
cd ../release/
pyinstaller -i ../data/time.ico -F ../5.func.py
pyinstaller -F ../0.test.py  # add libs 
