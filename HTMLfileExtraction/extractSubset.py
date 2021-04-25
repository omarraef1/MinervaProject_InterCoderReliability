# Omar R.G.

import json
#!pip install simplejson
#!pip install tqdm

import tarfile

import glob
import subprocess
import os
import zipfile
import tarfile

import simplejson as json
import shutil
from tqdm.notebook import tqdm as tqdm
import pandas as pd
import sys



def main():

    csvFile = pd.read_csv('subset_list_GSR.csv', encoding='utf-8')
    cols = ['file_name']
    nxtcols = csvFile
    nxtcols['tags'] = csvFile[cols].apply(lambda x: x.str.strip(), axis=1)
    nxtcols = nxtcols.drop(cols, axis = 1)
    nxtcols['tags'] = nxtcols.tags.str.split(";", expand=False)
    #print(len(nxtcols['tags']))

    fileLst = []
    for element in nxtcols['tags']:
        nu = element[0].split("-")
        nuu = nu[-1]
        nuuu = nuu.split(".")
        nuuuu = nuuu[0]
        #print(nuuuu)
        fileLst.append(nuuuu)
    #print(len(fileLst))
    
    for filename in glob.glob('./pool/*.plain.html'):
        fileNameSplit = filename.split("-")
        fnFiltered = fileNameSplit[-1]
        fnFur = fnFiltered.split(".")
        fnFin = fnFur[0]
        if(fnFin in fileLst):
            shutil.copy(filename, './extraction')
            #print(filename)
            
        #print(fnFin)





    
if __name__ == "__main__":
    main()
