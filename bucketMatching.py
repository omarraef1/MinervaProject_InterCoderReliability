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

# Sources similarities

    coder_1 = pd.read_csv('./pandas/output_ORG.csv', encoding='utf-8')
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


    coder_2 = pd.read_csv('./pandas/output_AK.csv', encoding='utf-8')
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

    coder_3 = pd.read_csv('./pandas/output_AAB.csv', encoding='utf-8')
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

    # article #2 = k = 1

    print("Coder_1 & Coder_2 (Cosine Similarities)")
    print()
    for k in range(499):
        print(" Article" + "     Sources" + "     Actions" + "  Targets" + "    Districts" + "    Provinces")
        article = k+1
        [totalSourcesCountedArticlesC1C2, totalSourcesPercantagesC1C2, srcPrcntg] = executeSimilarityCheck_Cosine(coder_1_Src['tags'][k], coder_2_Src['tags'][k],totalSourcesCountedArticlesC1C2, totalSourcesPercantagesC1C2)
        [totalActionsCountedArticlesC1C2, totalActionsPercantagesC1C2, actPrcntg] = executeSimilarityCheck_Cosine(coder_1_Act['tags'][k], coder_2_Act['tags'][k],totalActionsCountedArticlesC1C2, totalActionsPercantagesC1C2)
        [totalTargetsCountedArticlesC1C2, totalTargetsPercantagesC1C2, tarPrcntg] = executeSimilarityCheck_Cosine(coder_1_Tar['tags'][k], coder_2_Tar['tags'][k],totalTargetsCountedArticlesC1C2, totalTargetsPercantagesC1C2)
        [totalDistrictsCountedArticlesC1C2, totalDistrictsPercantagesC1C2, distPrcntg] = executeSimilarityCheck_Cosine(coder_1_Dist['tags'][k], coder_2_Dist['tags'][k],totalDistrictsCountedArticlesC1C2, totalDistrictsPercantagesC1C2)
        [totalProvincesCountedArticlesC1C2, totalProvincesPercantagesC1C2, provPrcntg] = executeSimilarityCheck_Cosine(coder_1_Prov['tags'][k], coder_2_Prov['tags'][k],totalProvincesCountedArticlesC1C2, totalProvincesPercantagesC1C2)
        
        print(str(article) + "  " + str(srcPrcntg) + "  " +
              str(actPrcntg)+"  "+
              str(tarPrcntg)+"  "+
              str(distPrcntg)+"  "+
              str(provPrcntg)+"  ")
        print()
        
    print()
    
    print("Valid Articles(Both coders annotated): " + str(totalSourcesCountedArticlesC1C2))
    #print(totalSourcesPercantagesC1C2)
    
    print("Overall Average Agreement:")
    c1c2Sources = totalSourcesPercantagesC1C2/totalSourcesCountedArticlesC1C2
    print("Coder_1 & Coder_2 (Sources): "+str(c1c2Sources))
    c1c2Actions = totalActionsPercantagesC1C2/totalActionsCountedArticlesC1C2
    print("Coder_1 & Coder_2 (Actions): "+str(c1c2Actions))
    c1c2Targets = totalTargetsPercantagesC1C2/totalTargetsCountedArticlesC1C2
    print("Coder_1 & Coder_2 (Targets): "+str(c1c2Targets))







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

    # article #2 = k = 1

    print("Coder_1 & Coder_3 (Cosine Similarities)")
    print()
    for k in range(499):
        print(" Article" + "     Sources" + "     Actions" + "  Targets" + "    Districts" + "    Provinces")
        article = k+1
        [totalSourcesCountedArticlesC1C3, totalSourcesPercantagesC1C3, srcPrcntg] = executeSimilarityCheck_Cosine(coder_1_Src['tags'][k], coder_3_Src['tags'][k],totalSourcesCountedArticlesC1C3, totalSourcesPercantagesC1C3)
        [totalActionsCountedArticlesC1C3, totalActionsPercantagesC1C3, actPrcntg] = executeSimilarityCheck_Cosine(coder_1_Act['tags'][k], coder_3_Act['tags'][k],totalActionsCountedArticlesC1C3, totalActionsPercantagesC1C3)
        [totalTargetsCountedArticlesC1C3, totalTargetsPercantagesC1C3, tarPrcntg] = executeSimilarityCheck_Cosine(coder_1_Tar['tags'][k], coder_3_Tar['tags'][k],totalTargetsCountedArticlesC1C3, totalTargetsPercantagesC1C3)
        [totalDistrictsCountedArticlesC1C3, totalDistrictsPercantagesC1C3, distPrcntg] = executeSimilarityCheck_Cosine(coder_1_Dist['tags'][k], coder_3_Dist['tags'][k],totalDistrictsCountedArticlesC1C3, totalDistrictsPercantagesC1C3)
        [totalProvincesCountedArticlesC1C3, totalProvincesPercantagesC1C3, provPrcntg] = executeSimilarityCheck_Cosine(coder_1_Prov['tags'][k], coder_3_Prov['tags'][k],totalProvincesCountedArticlesC1C3, totalProvincesPercantagesC1C3)
        
        print(str(article) + "  " + str(srcPrcntg) + "  " +
              str(actPrcntg)+"  "+
              str(tarPrcntg)+"  "+
              str(distPrcntg)+"  "+
              str(provPrcntg)+"  ")
        print()
        
    print()
    
    print("Valid Articles(Both coders annotated): " + str(totalSourcesCountedArticlesC1C3))
    #print(totalSourcesPercantagesC1C2)
    
    print("Overall Average Agreement:")
    c1c3Sources = totalSourcesPercantagesC1C3/totalSourcesCountedArticlesC1C3
    print("Coder_1 & Coder_3 (Sources): "+str(c1c3Sources))
    c1c3Actions = totalActionsPercantagesC1C3/totalActionsCountedArticlesC1C3
    print("Coder_1 & Coder_3 (Actions): "+str(c1c3Actions))
    c1c3Targets = totalTargetsPercantagesC1C3/totalTargetsCountedArticlesC1C3
    print("Coder_1 & Coder_3 (Targets): "+str(c1c3Targets))



    



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

    # article #2 = k = 1

    print("Coder_2 & Coder_3 (Cosine Similarities)")
    print()
    for k in range(499):
        print(" Article" + "     Sources" + "     Actions" + "  Targets" + "    Districts" + "    Provinces")
        article = k+1
        [totalSourcesCountedArticlesC2C3, totalSourcesPercantagesC2C3, srcPrcntg] = executeSimilarityCheck_Cosine(coder_2_Src['tags'][k], coder_3_Src['tags'][k],totalSourcesCountedArticlesC2C3, totalSourcesPercantagesC2C3)
        [totalActionsCountedArticlesC2C3, totalActionsPercantagesC2C3, actPrcntg] = executeSimilarityCheck_Cosine(coder_2_Act['tags'][k], coder_3_Act['tags'][k],totalActionsCountedArticlesC2C3, totalActionsPercantagesC2C3)
        [totalTargetsCountedArticlesC2C3, totalTargetsPercantagesC2C3, tarPrcntg] = executeSimilarityCheck_Cosine(coder_2_Tar['tags'][k], coder_3_Tar['tags'][k],totalTargetsCountedArticlesC2C3, totalTargetsPercantagesC2C3)
        [totalDistrictsCountedArticlesC2C3, totalDistrictsPercantagesC2C3, distPrcntg] = executeSimilarityCheck_Cosine(coder_2_Dist['tags'][k], coder_3_Dist['tags'][k],totalDistrictsCountedArticlesC2C3, totalDistrictsPercantagesC2C3)
        [totalProvincesCountedArticlesC2C3, totalProvincesPercantagesC2C3, provPrcntg] = executeSimilarityCheck_Cosine(coder_2_Prov['tags'][k], coder_3_Prov['tags'][k],totalProvincesCountedArticlesC2C3, totalProvincesPercantagesC2C3)
        
        print(str(article) + "  " + str(srcPrcntg) + "  " +
              str(actPrcntg)+"  "+
              str(tarPrcntg)+"  "+
              str(distPrcntg)+"  "+
              str(provPrcntg)+"  ")
        print()
        
    print()
    
    print("Valid Articles(Both coders annotated): " + str(totalSourcesCountedArticlesC2C3))
    #print(totalSourcesPercantagesC1C2)
    
    print("Overall Average Agreement:")
    c2c3Sources = totalSourcesPercantagesC2C3/totalSourcesCountedArticlesC2C3
    print("Coder_2 & Coder_3 (Sources): "+str(c2c3Sources))
    c2c3Actions = totalActionsPercantagesC2C3/totalActionsCountedArticlesC2C3
    print("Coder_2 & Coder_3 (Actions): "+str(c2c3Actions))
    c2c3Targets = totalTargetsPercantagesC2C3/totalTargetsCountedArticlesC2C3
    print("Coder_2 & Coder_3 (Targets): "+str(c2c3Targets))




    #executeSimilarityCheck_Jaccard(coder_1['tags'][k], coder_3['tags'][k])
    #print()
    

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
    #print(a)
    #print(b)
    counterA = Counter(a)
    counterB = Counter(b)
    if(len(counterA)==0 or len(counterB)==0):
        return countedArticle, percYield, 0.0

    result = counter_cosine_similarity(counterA, counterB)
    countedArticle += 1
    percYield += result
    return countedArticle, percYield, result


def executeSimilarityCheck_Jaccard(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return (len(s1.intersection(s2)) / len(s1.union(s2)))




'''
    print(a)
    print(b)
    print("Cosine Similarity: " + td.cosine.normalized_similarity(a,b))
    print("Jaccard Similarity: " + td.jaccard.normalized_similarity(a,b))
    print("Sorensen Similarity: " + td.sorensen.normalized_similarity(a,b))
    print("Levenshtein Similarity: " + td.levenshtein.normalized_similarity(a,b))
'''

    
if __name__ == "__main__":
    main()

