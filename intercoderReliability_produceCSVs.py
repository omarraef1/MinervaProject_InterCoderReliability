# Omar R. Gebril

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

    classIds_AK = {"e_1": "Target",
                "e_10": "Actor_Filter",
                "e_2": "Action",
                "e_3": "District",
                "e_4": "Location_Filter",
                "e_5": "Province",
                "e_6": "Source",
                "e_9": "Action_Filter"}
    
    classIds_ORG = {"e_1": "Target",
                "e_10": "Actor_Filter",
                "e_2": "Action",
                "e_3": "District",
                "e_4": "Location_Filter",
                "e_5": "Province",
                "e_6": "Source",
                "e_9": "Action_Filter"}
    
    classIds_AAB = {"e_1": "Source",
                "e_10": "Actor_Filter",
                "e_2": "Target",
                "e_3": "Action",
                "e_6": "Location_Filter",
                "e_7": "District",
                "e_8": "Province",
                "e_9": "Action_Filter"}

    #totalAnnFiles = len(glob.glob('./extract_AK/akArticles_Ann/*.json'))
    #print(totalAnnFiles)

    dataPandas = []
    for filename in glob.glob('./extract_AK/akArticles_Ann/*.json'):
        Sources = []
        Actions = []
        Targets = []
        Districts = []
        Provinces = []

        filenameSplit = filename.split("-")
        filenameFilteredExtension = filenameSplit[-1].split(".")
        newFilename = str(filenameFilteredExtension[0]) + ".txt"

        currFile = open(filename,encoding="utf-8", errors='ignore')

        ffdata = json.load(currFile)
        setup_Pandas(ffdata,classIds_AK, Sources, Actions, Targets, Districts, Provinces, newFilename, dataPandas)
        currFile.close()
        
    df = pd.DataFrame(dataPandas)
    df.to_csv('./pandas/output_AK.csv',encoding= 'utf-8')

    dataPandas = [] 
    for filename in glob.glob('./extract_ORG/orgArticles_Ann/*.json'):
        Sources = []
        Actions = []
        Targets = []
        Districts = []
        Provinces = []

        filenameSplit = filename.split("-")
        filenameFilteredExtension = filenameSplit[-1].split(".")
        newFilename = str(filenameFilteredExtension[0]) + ".txt"

        currFile = open(filename,encoding="utf-8", errors='ignore')

        ffdata = json.load(currFile)
        setup_Pandas(ffdata,classIds_ORG, Sources, Actions, Targets, Districts, Provinces, newFilename, dataPandas)
        currFile.close()
        
    df = pd.DataFrame(dataPandas)
    df.to_csv('./pandas/output_ORG.csv',encoding= 'utf-8')

    dataPandas = []
    for filename in glob.glob('./extract_AAB/aabArticles_Ann/*.json'):
        Sources = []
        Actions = []
        Targets = []
        Districts = []
        Provinces = []

        filenameSplit = filename.split("-")
        filenameFilteredExtension = filenameSplit[-1].split(".")
        newFilename = str(filenameFilteredExtension[0]) + ".txt"

        currFile = open(filename,encoding="utf-8", errors='ignore')

        ffdata = json.load(currFile)
        setup_Pandas(ffdata,classIds_AAB, Sources, Actions, Targets, Districts, Provinces, newFilename, dataPandas)
        currFile.close()
        
    df = pd.DataFrame(dataPandas)
    df.to_csv('./pandas/output_AAB.csv',encoding= 'utf-8')

    print("CSVs Produced, Program Terminated. Cheers!")

def setup_Pandas(fJsonData, classIds, sources, actions, targets, districts, provinces, filename, dataForPanda):
    
    entities = fJsonData["entities"]

    if(len(entities)>0):
        #classId = entities[0]["classId"] #first annotation
        #parPart = entities[0]["part"]
        #offsets = entities[0]["offsets"][0]
        #textStart = offsets["start"]
        #text = offsets["text"]

        for entity in entities:
            entityDict = {"Part":entity["part"],"Type": entity["classId"], "Text": entity["offsets"][0]["text"],
                          "Start":entity["offsets"][0]["start"],"End":(len(entity["offsets"][0]["text"])-1+entity["offsets"][0]["start"])}
            if (classIds[entity["classId"]] == "Source"):
                sources.append(entity["offsets"][0]["text"])
            elif(classIds[entity["classId"]] == "Target"):
                targets.append(entity["offsets"][0]["text"])
            elif(classIds[entity["classId"]] == "Action"):
                actions.append(entity["offsets"][0]["text"])
            elif(classIds[entity["classId"]] == "District"):
                districts.append(entity["offsets"][0]["text"])
            elif(classIds[entity["classId"]] == "Province"):
                provinces.append(entity["offsets"][0]["text"])

        if(len(sources)>0):
            sourceTup = tuple(sources)
        else:
            sourceTup = ""
            
        if(len(actions)>0):
            actionTup = tuple(actions)
        else:
            actionTup = ""
        
        if(len(targets)>0):
            targetTup = tuple(targets)
        else:
            targetTup = ""
        
        if(len(districts)>0):
            districtTup = tuple(districts)
        else:
            districtTup = ""
        
        if(len(provinces)>0):
            provinceTup = tuple(provinces)
        else:
            provinceTup = ""
            
        #sourceTup = tuple(sources)
        #actionTup = tuple(actions)
        #targetTup = tuple(targets)
        #districtTup = tuple(districts)
        #provinceTup = tuple(provinces)

        dataForPanda.append({"File": str(filename), "Sources": sourceTup, "Actions": actionTup, "Targets": targetTup, "Districts": districtTup, "Provinces": provinceTup})
         
    else:
        #add empty record
        dataForPanda.append({"File": str(filename), "Sources": "", "Actions": "", "Targets": "", "Districts": "", "Provinces": ""})

    #df = pd.DataFrame(dataForPanda)



if __name__ == "__main__":
    main()
