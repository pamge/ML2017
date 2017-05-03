#!/bin/bash

# python 2.7
wget 'https://www.csie.ntu.edu.tw/~r05922085/best_model'
python3 hw3.py test $1 $2
