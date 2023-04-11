import pandas as pd
import pathlib
import os
import numpy as np
import ast
import sys
import math



path_to_csv = "C:/Users/marcio/Videos/Ian_videos/results/wave/wave_corrected__20191113_1401_58-2_L_A.csv" #sys.argv[1]
path_to_save = "C:/Users/marcio/Videos/Ian_videos/results/cycle" # sys.argv[2]

final_path = pathlib.PurePath(path_to_csv)
expe = final_path.name[16:][:-4]

if os.path.exists(path_to_save + "/wave_cycle_" + expe + ".csv"):
    print("CSV resulting file exist, remove it first")
    quit()
else:
    print("CSV resulting file does not exist, it will be created")

df = pd.read_csv(path_to_csv)

df['angle_corr_tail'] = df['angle_corr_tail'].apply(ast.literal_eval)


df["ant"] = df["angle_corr_tail"].apply(lambda x: tuple(zip(*x))[1][-3])
df["pen"] = df["angle_corr_tail"].apply(lambda x: tuple(zip(*x))[1][-2])
df["ult"] = df["angle_corr_tail"].apply(lambda x: tuple(zip(*x))[1][-1])






df["cycle"] = np.NAN
sequnces_list = df.sequence.unique()
for n in sequnces_list:    
    the_idxs = df.loc[df["sequence"] == n].index      
    cycle_number = 1
    count=0  
    for i in the_idxs:       
        df.loc[(df.index == i) & (df["sequence"] == n), "cycle"] = cycle_number
        count=count+1        
        if count == 6:
            cycle_number=cycle_number+1
            count = 0 
                





'''
df["cycle"] = np.NAN
sequnces_list = df["sequence"].unique()
for p in sequnces_list:
    cycle_number = 1
    
    the_idxs = df.loc[(df["sequence"] == p)].index
    
    for real_index, nominal_index in enumerate(the_idxs):
               
        current_position = df.loc[(df.index == nominal_index), "ult"].iloc[0]
            
        if real_index == 0:
            last_position = current_position
                    
        elif real_index == 1:           
            if current_position > last_position:                
                trend = "up"
                
            elif current_position < last_position:
                trend = "down"
                
            else:
                trend = "undefined"
                #  Keep the same last_tail_direction                
                            
        else:
            if (current_position > last_position) & (trend=="undefined"):
                trend = "up"
                
            elif (current_position < last_position) & (trend=="undefined"):
                trend = "down"
                
            elif (current_position > last_position) & (trend=="dow"):
                cycle_number=cycle_number+1
            
            elif (current_position < last_position) & (trend=="up"):
                cycle_number=cycle_number+1             

            elif (current_position == last_position):
                pass
            
        

        last_position = current_position
        
        df.loc[(df.index == nominal_index), "cycle"] = cycle_number    
    

'''



df['diffs'] = df.groupby(["sequence","cycle"])['ult'].diff()


df['center_of_mass'] = df['center_of_mass'].apply(ast.literal_eval)


temp = df.groupby(["sequence", "cycle"])
the_firsts = temp.head(1)
the_firsts = the_firsts.sort_values(by=["sequence", "cycle"])   
the_lasts = temp.tail(1)
the_lasts = the_lasts.sort_values(by=["sequence", "cycle"]) 
the_firsts["distance_cycle"] = np.NaN
print(the_firsts.columns)

for i in range(0, len(the_firsts)):
    first_com = the_firsts.iloc[i, 2]    
    last_com = the_lasts.iloc[i, 2]    
    the_firsts.iloc[i, 24] = math.hypot((first_com[0] - last_com[0]), (first_com[1] - last_com[1]))      
    
    
temp = the_firsts[["sequence", "cycle", "distance_cycle"]]

df = df.merge(temp, on=["sequence", "cycle"])


df.to_csv(path_to_save + '/wave_cycle_' + expe + ".csv", mode='w', index=True, header=True)





