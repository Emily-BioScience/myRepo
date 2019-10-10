#!/usr/bin/env bash
# input: ../data/3rd-meeting.ALT-AST.data.txt
cd bin/
python 3.rank.py    # => ../output/3rd-meeting.processedData.txt
Rscript 3.rank.r    # => ../output/3rd-meeting.summaryTable.txt
Rscript 3.select.r  # => ../output/3rd-meeting.allAnimals.txt, ../output/3rd-meeting.corrplot.pdf, ../output/3rd-meeting.num_cut1.low_cut4.high_cut7.kmeans.txt


# 2nd input: ../data/2019.09.rawData.txt
cd bin/
python 3.rank.py   # => ../output/2019.09.summaryTable.txt, ../output/2019.09.allSamples.txt, ../output/2019.09.selectSamples.txt
Rscript 3.kmeans.r # => ../output/2019.09.selectSamples-cluster.txt
