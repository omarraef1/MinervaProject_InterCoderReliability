# Omar R. Gebril

from collections import Counter
import math
import textdistance as td
import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer


def main():

    #Create buckets-Start
    
    coder_1 = pd.read_csv('output_ORG_iter1.csv', encoding='utf-8')
    #cols = ['Sources', 'Actions', 'Targets', 'Districts', 'Provinces']
    cols_Src = ['Sources']
    coder_1_Src = coder_1
    coder_1_Src['tags'] = coder_1[cols_Src].apply(lambda x: x.str.strip(' ()').str.cat(sep=',').replace('\'',''), axis=1)
    coder_1_Src = coder_1_Src.drop(cols_Src, axis=1)
    coder_1_Src['tags'] = coder_1_Src.tags.str.split(",", expand=False)

    cols_Act = ['Actions']
    coder_1_Act = coder_1
    coder_1_Act['tags'] = coder_1[cols_Act].apply(lambda x: x.str.strip(' ()').str.cat(sep=',').replace('\'',''), axis=1)
    coder_1_Act = coder_1_Act.drop(cols_Act, axis=1)
    coder_1_Act['tags'] = coder_1_Act.tags.str.split(",", expand=False)

    cols_Tar = ['Targets']
    coder_1_Tar = coder_1
    coder_1_Tar['tags'] = coder_1[cols_Tar].apply(lambda x: x.str.strip(' ()').str.cat(sep=',').replace('\'',''), axis=1)
    coder_1_Tar = coder_1_Tar.drop(cols_Tar, axis=1)
    coder_1_Tar['tags'] = coder_1_Tar.tags.str.split(",", expand=False)
    
    cols_Dist = ['Districts']
    coder_1_Dist = coder_1
    coder_1_Dist['tags'] = coder_1[cols_Dist].apply(lambda x: x.str.strip(' ()').str.cat(sep=',').replace('\'',''), axis=1)
    coder_1_Dist = coder_1_Dist.drop(cols_Dist, axis=1)
    coder_1_Dist['tags'] = coder_1_Dist.tags.str.split(",", expand=False)
    
    cols_Prov = ['Provinces']
    coder_1_Prov = coder_1
    coder_1_Prov['tags'] = coder_1[cols_Prov].apply(lambda x: x.str.strip(' ()').str.cat(sep=',').replace('\'',''), axis=1)
    coder_1_Prov = coder_1_Prov.drop(cols_Prov, axis=1)
    coder_1_Prov['tags'] = coder_1_Prov.tags.str.split(",", expand=False)
    #print(len(coder_1['tags'])) #499
    #print(coder_1['tags'])


    coder_2 = pd.read_csv('output_AK_iter1.csv', encoding='utf-8')
    #cols = ['Sources', 'Actions', 'Targets', 'Districts', 'Provinces']
    cols_Src = ['Sources']
    coder_2_Src = coder_2
    coder_2_Src['tags'] = coder_2[cols_Src].apply(lambda x: x.str.strip(' ()').str.cat(sep=',').replace('\'',''), axis=1)
    coder_2_Src = coder_2_Src.drop(cols_Src, axis=1)
    coder_2_Src['tags'] = coder_2_Src.tags.str.split(",", expand=False)

    cols_Act = ['Actions']
    coder_2_Act = coder_2
    coder_2_Act['tags'] = coder_2[cols_Act].apply(lambda x: x.str.strip(' ()').str.cat(sep=',').replace('\'',''), axis=1)
    coder_2_Act = coder_2_Act.drop(cols_Act, axis=1)
    coder_2_Act['tags'] = coder_2_Act.tags.str.split(",", expand=False)

    cols_Tar = ['Targets']
    coder_2_Tar = coder_2
    coder_2_Tar['tags'] = coder_2[cols_Tar].apply(lambda x: x.str.strip(' ()').str.cat(sep=',').replace('\'',''), axis=1)
    coder_2_Tar = coder_2_Tar.drop(cols_Tar, axis=1)
    coder_2_Tar['tags'] = coder_2_Tar.tags.str.split(",", expand=False)
    
    cols_Dist = ['Districts']
    coder_2_Dist = coder_2
    coder_2_Dist['tags'] = coder_2[cols_Dist].apply(lambda x: x.str.strip(' ()').str.cat(sep=',').replace('\'',''), axis=1)
    coder_2_Dist = coder_2_Dist.drop(cols_Dist, axis=1)
    coder_2_Dist['tags'] = coder_2_Dist.tags.str.split(",", expand=False)
    
    cols_Prov = ['Provinces']
    coder_2_Prov = coder_2
    coder_2_Prov['tags'] = coder_2[cols_Prov].apply(lambda x: x.str.strip(' ()').str.cat(sep=',').replace('\'',''), axis=1)
    coder_2_Prov = coder_2_Prov.drop(cols_Prov, axis=1)
    coder_2_Prov['tags'] = coder_2_Prov.tags.str.split(",", expand=False)
    #print(len(coder_1['tags'])) #499
    #print(coder_1['tags'])

    coder_3 = pd.read_csv('output_AAB_iter1.csv', encoding='utf-8')
    #cols = ['Sources', 'Actions', 'Targets', 'Districts', 'Provinces']
    cols_Src = ['Sources']
    coder_3_Src = coder_3
    coder_3_Src['tags'] = coder_3[cols_Src].apply(lambda x: x.str.strip(' ()').str.cat(sep=',').replace('\'',''), axis=1)
    coder_3_Src = coder_3_Src.drop(cols_Src, axis=1)
    coder_3_Src['tags'] = coder_3_Src.tags.str.split(",", expand=False)

    cols_Act = ['Actions']
    coder_3_Act = coder_3
    coder_3_Act['tags'] = coder_3[cols_Act].apply(lambda x: x.str.strip(' ()').str.cat(sep=',').replace('\'',''), axis=1)
    coder_3_Act = coder_3_Act.drop(cols_Act, axis=1)
    coder_3_Act['tags'] = coder_3_Act.tags.str.split(",", expand=False)

    cols_Tar = ['Targets']
    coder_3_Tar = coder_3
    coder_3_Tar['tags'] = coder_3[cols_Tar].apply(lambda x: x.str.strip(' ()').str.cat(sep=',').replace('\'',''), axis=1)
    coder_3_Tar = coder_3_Tar.drop(cols_Tar, axis=1)
    coder_3_Tar['tags'] = coder_3_Tar.tags.str.split(",", expand=False)
    
    cols_Dist = ['Districts']
    coder_3_Dist = coder_3
    coder_3_Dist['tags'] = coder_3[cols_Dist].apply(lambda x: x.str.strip(' ()').str.cat(sep=',').replace('\'',''), axis=1)
    coder_3_Dist = coder_3_Dist.drop(cols_Dist, axis=1)
    coder_3_Dist['tags'] = coder_3_Dist.tags.str.split(",", expand=False)
    
    cols_Prov = ['Provinces']
    coder_3_Prov = coder_3
    coder_3_Prov['tags'] = coder_3[cols_Prov].apply(lambda x: x.str.strip(' ()').str.cat(sep=',').replace('\'',''), axis=1)
    coder_3_Prov = coder_3_Prov.drop(cols_Prov, axis=1)
    coder_3_Prov['tags'] = coder_3_Prov.tags.str.split(",", expand=False)
    #print(len(coder_1['tags'])) #499
    #print(coder_1['tags'])

    #Create_Buckets--END

    # Coder_1 & Coder_2 (Cosine similarities)
    
    totalSourcesCountedArticlesC1C2 = 0.0
    totalSourcesPercantagesC1C2 = 0.0
    totalActionsCountedArticlesC1C2 = 0.0
    totalActionsPercantagesC1C2 = 0.0
    totalTargetsCountedArticlesC1C2 = 0.0
    totalTargetsPercantagesC1C2 = 0.0
    totalDistrictsCountedArticlesC1C2 = 0.0
    totalDistrictsPercantagesC1C2 = 0.0
    totalProvincesCountedArticlesC1C2 = 0.0
    totalProvincesPercantagesC1C2 = 0.0

    # Coder_1 & Coder_3 (Cosine similarities)
    
    totalSourcesCountedArticlesC1C3 = 0.0
    totalSourcesPercantagesC1C3 = 0.0
    totalActionsCountedArticlesC1C3 = 0.0
    totalActionsPercantagesC1C3 = 0.0
    totalTargetsCountedArticlesC1C3 = 0.0
    totalTargetsPercantagesC1C3 = 0.0
    totalDistrictsCountedArticlesC1C3 = 0.0
    totalDistrictsPercantagesC1C3 = 0.0
    totalProvincesCountedArticlesC1C3 = 0.0
    totalProvincesPercantagesC1C3 = 0.0

    # Coder_2 & Coder_3 (Cosine similarities)
    
    totalSourcesCountedArticlesC2C3 = 0.0
    totalSourcesPercantagesC2C3 = 0.0
    totalActionsCountedArticlesC2C3 = 0.0
    totalActionsPercantagesC2C3 = 0.0
    totalTargetsCountedArticlesC2C3 = 0.0
    totalTargetsPercantagesC2C3 = 0.0
    totalDistrictsCountedArticlesC2C3 = 0.0
    totalDistrictsPercantagesC2C3 = 0.0
    totalProvincesCountedArticlesC2C3 = 0.0
    totalProvincesPercantagesC2C3 = 0.0

    dataPandas = []

    for k in range(499): #499 to 99
        article = k+1
        #[totalSourcesCountedArticlesC1C2, totalSourcesPercantagesC1C2, srcPrcntg12] = executeSimilarityCheck_Cosine(coder_1_Src['tags'][k], coder_2_Src['tags'][k],totalSourcesCountedArticlesC1C2, totalSourcesPercantagesC1C2)
        #[totalActionsCountedArticlesC1C2, totalActionsPercantagesC1C2, actPrcntg12] = executeSimilarityCheck_Cosine(coder_1_Act['tags'][k], coder_2_Act['tags'][k],totalActionsCountedArticlesC1C2, totalActionsPercantagesC1C2)
        #[totalTargetsCountedArticlesC1C2, totalTargetsPercantagesC1C2, tarPrcntg12] = executeSimilarityCheck_Cosine(coder_1_Tar['tags'][k], coder_2_Tar['tags'][k],totalTargetsCountedArticlesC1C2, totalTargetsPercantagesC1C2)
        #[totalDistrictsCountedArticlesC1C2, totalDistrictsPercantagesC1C2, distPrcntg12] = executeSimilarityCheck_Cosine(coder_1_Dist['tags'][k], coder_2_Dist['tags'][k],totalDistrictsCountedArticlesC1C2, totalDistrictsPercantagesC1C2)
        #[totalProvincesCountedArticlesC1C2, totalProvincesPercantagesC1C2, provPrcntg12] = executeSimilarityCheck_Cosine(coder_1_Prov['tags'][k], coder_2_Prov['tags'][k],totalProvincesCountedArticlesC1C2, totalProvincesPercantagesC1C2)

        [totalSourcesCountedArticlesC1C3, totalSourcesPercantagesC1C3, srcPrcntg13] = executeSimilarityCheck_Cosine(coder_1_Src['tags'][k], coder_3_Src['tags'][k],totalSourcesCountedArticlesC1C3, totalSourcesPercantagesC1C3)
        [totalActionsCountedArticlesC1C3, totalActionsPercantagesC1C3, actPrcntg13] = executeSimilarityCheck_Cosine(coder_1_Act['tags'][k], coder_3_Act['tags'][k],totalActionsCountedArticlesC1C3, totalActionsPercantagesC1C3)
        [totalTargetsCountedArticlesC1C3, totalTargetsPercantagesC1C3, tarPrcntg13] = executeSimilarityCheck_Cosine(coder_1_Tar['tags'][k], coder_3_Tar['tags'][k],totalTargetsCountedArticlesC1C3, totalTargetsPercantagesC1C3)
        [totalDistrictsCountedArticlesC1C3, totalDistrictsPercantagesC1C3, distPrcntg13] = executeSimilarityCheck_Cosine(coder_1_Dist['tags'][k], coder_3_Dist['tags'][k],totalDistrictsCountedArticlesC1C3, totalDistrictsPercantagesC1C3)
        [totalProvincesCountedArticlesC1C3, totalProvincesPercantagesC1C3, provPrcntg13] = executeSimilarityCheck_Cosine(coder_1_Prov['tags'][k], coder_3_Prov['tags'][k],totalProvincesCountedArticlesC1C3, totalProvincesPercantagesC1C3)

        #[totalSourcesCountedArticlesC2C3, totalSourcesPercantagesC2C3, srcPrcntg23] = executeSimilarityCheck_Cosine(coder_2_Src['tags'][k], coder_3_Src['tags'][k],totalSourcesCountedArticlesC2C3, totalSourcesPercantagesC2C3)
        #[totalActionsCountedArticlesC2C3, totalActionsPercantagesC2C3, actPrcntg23] = executeSimilarityCheck_Cosine(coder_2_Act['tags'][k], coder_3_Act['tags'][k],totalActionsCountedArticlesC2C3, totalActionsPercantagesC2C3)
        #[totalTargetsCountedArticlesC2C3, totalTargetsPercantagesC2C3, tarPrcntg23] = executeSimilarityCheck_Cosine(coder_2_Tar['tags'][k], coder_3_Tar['tags'][k],totalTargetsCountedArticlesC2C3, totalTargetsPercantagesC2C3)
        #[totalDistrictsCountedArticlesC2C3, totalDistrictsPercantagesC2C3, distPrcntg23] = executeSimilarityCheck_Cosine(coder_2_Dist['tags'][k], coder_3_Dist['tags'][k],totalDistrictsCountedArticlesC2C3, totalDistrictsPercantagesC2C3)
        #[totalProvincesCountedArticlesC2C3, totalProvincesPercantagesC2C3, provPrcntg23] = executeSimilarityCheck_Cosine(coder_2_Prov['tags'][k], coder_3_Prov['tags'][k],totalProvincesCountedArticlesC2C3, totalProvincesPercantagesC2C3)

        #print(totPerc)
        #print(percCount)
        #print()

     #   if(percCount==0):
      #      avg_agreement = 0
       # else:
        #    avg_agreement = totPerc / percCount
        #
        #avg_agreement = (srcPrcntg12 + srcPrcntg13 + srcPrcntg23 + actPrcntg12 + actPrcntg13 + actPrcntg23 + tarPrcntg12 + tarPrcntg13 + tarPrcntg23 + distPrcntg12 + distPrcntg13 + distPrcntg23 + provPrcntg12 + provPrcntg13 + provPrcntg23 ) / 15
        
        totPerc = 0
        percCount = 0

        if (srcPrcntg13!=-1.0):
            totPerc+=srcPrcntg13
            percCount+=1
        if (actPrcntg13!=-1.0):
            totPerc+=actPrcntg13
            percCount+=1
        if (tarPrcntg13!=-1.0):
            totPerc+=tarPrcntg13
            percCount+=1
        if (distPrcntg13!=-1.0):
            totPerc+=distPrcntg13
            percCount+=1
        if (provPrcntg13!=-1.0):
            totPerc+=provPrcntg13
            percCount+=1

        if(percCount==0):
            avg_agreement = -1
        else:
            avg_agreement = totPerc / percCount
        #avg_agreement = (srcPrcntg13 + actPrcntg13 + tarPrcntg13 + distPrcntg13 + provPrcntg13 ) / 5

      #  dataPandas.append({"file_name": article, "sources_c1&c2": srcPrcntg12, "sources_c1&c3": srcPrcntg13, "sources_c2&c3": srcPrcntg23,
       #                    "actions_c1&c2": actPrcntg12, "actions_c1&c3": actPrcntg13, "actions_c2&c3": actPrcntg23,
        #                   "targets_c1&c2": tarPrcntg12, "targets_c1&c3": tarPrcntg13, "targets_c2&c3": tarPrcntg23,
         #                  "districts_c1&c2": distPrcntg12, "districts_c1&c3": distPrcntg13, "districts_c2&c3": distPrcntg23,
          #                 "provinces_c1&c2": provPrcntg12, "provinces_c1&c3": provPrcntg13, "provinces_c2&c3": provPrcntg23, "average_agreement": avg_agreement})

        dataPandas.append({"file_name": article, "sources_c1&c3": srcPrcntg13, "actions_c1&c3": actPrcntg13, "targets_c1&c3": tarPrcntg13,
                           "districts_c1&c3": distPrcntg13, "provinces_c1&c3": provPrcntg13, "average_agreement": avg_agreement})
    
        df = pd.DataFrame(dataPandas)
        df.to_csv('cosineSimilarities_iter1.csv', encoding = 'utf-8')

    

    

def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)


def executeSimilarityCheck_Cosine(a, b, countedArticle, percYield):
    for i in range(len(a)):
        if(len(a[i])==0):
            a.remove(a[i])
    
    for i in range(len(b)):
        if(len(b[i])==0):
            b.remove(b[i])

    
    if(len(a)==0 and len(b)==0):
        return countedArticle, percYield, -1.0 # if both have no annotations, return -1: ignor later
    if(len(a)==0 or len(b)==0):
        return countedArticle, percYield, 0.0 # if one has no annotations, return 0: they do not agree

    c = list(set(a))
    d = list(set(b))
    #print(c)
    #print(d)
    
    totalAnns = len(c) + len(d)
    simCount = 0

    if(len(c)<len(d)):
        for i in range(len(c)):
            firstflag = 1
            for j in range(len(d)):
                counterA = Counter(c[i])
                counterB = Counter(d[j])
                tmpSim = counter_cosine_similarity(counterA, counterB)
                if(tmpSim >= 0.8):
                    if(firstflag==1):
                        simCount+=2
                        firstflag=0
                    else:
                        simCount+=1
        if (simCount > totalAnns):
            simCount = totalAnns
        result = simCount / totalAnns
        countedArticle += 1
        percYield += result
        return countedArticle, percYield, result
    else:
        for i in range(len(d)):
            firstflag = 1
            for j in range(len(c)):
                counterB = Counter(d[i])
                counterA = Counter(c[j])
                tmpSim = counter_cosine_similarity(counterB, counterA)
                if(tmpSim >= 0.8):
                    if(firstflag==1):
                        simCount+=2
                        firstflag=0
                    else:
                        simCount+=1
        if (simCount > totalAnns):
            simCount = totalAnns
        result = simCount / totalAnns
        countedArticle += 1
        percYield += result
        #print(simCount)
        #print(totalAnns)
        return countedArticle, percYield, result






    
if __name__ == "__main__":
    main()
