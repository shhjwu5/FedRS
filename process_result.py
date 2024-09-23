# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt

base_path = "./baselines"
files = os.listdir(base_path)
# print(files)

all_results = pd.DataFrame(columns=['Name','Converge','Loss','Hitratio',"NDCG","Compressionratio","Stdtime"])

for i,file_name in enumerate(files):
    file_path = base_path+"/"+file_name
    print(file_name)
    f = open(file_path,"r")
    lines = f.readlines()
    f.close()
    
    result = {'Name':file_name,
              'Converge':1e5,
              'Loss':1e5,
              'Hitratio':0,
              'NDCG':0,
              'Compressionratio':0,
              'Stdtime':0,
              }
    
    line_num = 0
    
    for i in range(50):
        while line_num<len(lines) and lines[line_num][:5] != "Epoch":
            line_num += 1
        if line_num>=len(lines):
            break
        epoch = int(lines[line_num].split()[-1])
        loss = float(lines[line_num+2].split()[-1])
        hitratio = float(lines[line_num+3].split()[-1])
        ndcg = float(lines[line_num+4].split()[-1])
        compress = float(lines[line_num+5].split()[-1])
        stdtime = float(lines[line_num+7].split()[-1])
        
        if loss<result["Loss"]:
            result["Converge"] = epoch
            result["Loss"] = loss
            result["Hitratio"] = hitratio
            result["NDCG"] = ndcg
        
        result["Compressionratio"] += compress
        result["Stdtime"] += stdtime
        
        line_num += 1
        
    result["Stdtime"] /= 50
    result["Compressionratio"] /= 50
    
    # if file_name=="nc10-miuc1-mauc10-ae50-le10-bs128-paadaptive-ptratio-cr0.7-bamomentum-r10.1-r20.1-ld32-sd0-sr2.out":
        # print(result)
        
    all_results.loc[len(all_results)] = result

all_results.to_csv("./SOTA.csv")