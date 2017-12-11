# -*- coding: utf-8 -*-
import sys, csv, io, re
import argparse
import jieba
import jieba.posseg
import numpy as np

def main(args):
    input_data = [args.data1, args.data2, args.data3, args.data4, args.data5]
    fout = open(args.output, 'w')
    jieba.set_dictionary("jieba_dict/dict.txt.big")
    for i in input_data:
        if i == None:
            continue
        with open(i, 'r') as f:
            for line in f:
                words = jieba.cut(line.strip('\n'), cut_all=False)
                for word in words:
                    fout.write(word+' ')
                fout.write('\n')
    fout.close()
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data1", help="input data1", type=str, required=True)
    parser.add_argument("--data2", help="input data2", type=str)
    parser.add_argument("--data3", help="input data3", type=str)
    parser.add_argument("--data4", help="input data4", type=str)
    parser.add_argument("--data5", help="input data5", type=str)
    parser.add_argument("--output", help="output file", type=str, required=True)
    
    args = parser.parse_args()

    main(args)
