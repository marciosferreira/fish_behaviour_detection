
import os
import pathlib
import sys

path_to_csv = "C:/Users/marcio/Videos/Ian_videos/croped_Ian/errors/20191111_1527_5-1_L_A.csv" #sys.argv[1]
path_to_save = "C:/Users/marcio/Videos/Ian_videos/croped_Ian/errors/" #sys.argv[2]

final_path = pathlib.PurePath(path_to_csv)
expe = final_path.name[:-4]

if os.path.exists(path_to_save + "/rotated_" + expe + ".csv"):
  #os.remove(path_to_save + "/" + expe + ".csv")
  print("CSV save file exist, remove it first run it again")
  quit()
else:
  print("CSV save file does not exist, it will be created")
  
quadrante_numbers = [0, 1, 2, 3] # 0 to 3
fish_identification = [1, 2] # 1 to 2
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import numpy as np

df = pd.read_csv(path_to_csv)

df = df.set_index(['frame_number'])

df["take"] = None

import ast
df['fish_head'] = df['fish_head'].apply(ast.literal_eval)
df['fish_tail'] = df['fish_tail'].apply(ast.literal_eval)
#df['fish_id'] = df['fish_id'].apply(ast.literal_eval)
df['tail_points'] = df['tail_points'].apply(ast.literal_eval)
df['quad_coord'] = df['quad_coord'].apply(ast.literal_eval)


set_number = 1
for quadrant in quadrante_numbers:    
    for fish_ident in fish_identification:        
        list_frames = list(df[(df["quadrant"] == quadrant) & ((df["fish_id"] == str(fish_ident)) | (df["fish_id"] == "X") | (df["fish_id"] == "Y")) ].index.unique())
        for idx, value in enumerate(list_frames):
            if idx > 0:
                if idx%100==0: 
                    print("quadrant:" , quadrant)
                    print("fish_id:", fish_ident)
                    print("the idx:", idx)
                the_value = df.loc[(df.index==value) & (df.quadrant == quadrant) & ((df["fish_id"] == str(fish_ident)) | (df["fish_id"] == "X") | (df["fish_id"] == "Y")), 'fish_id'].iloc[0]
                previous_value = df.loc[(df.index==list_frames[idx-1]) & (df.quadrant == quadrant) & ((df["fish_id"] == str(fish_ident)) | (df["fish_id"] == "X") | (df["fish_id"] == "Y")), 'fish_id'].iloc[0]
                if the_value.isnumeric() and previous_value.isnumeric():
                    df.loc[(df.index==value) & (df.quadrant == quadrant) & (df["fish_id"] == str(fish_ident)), "take"] = set_number
                    
                elif not the_value.isnumeric() and (previous_value.isnumeric()):
                    set_number+=1
                            
                elif the_value.isnumeric() and not (previous_value.isnumeric()):
                    df.loc[(df.index==value) & (df.quadrant == quadrant) & (df["fish_id"] == str(fish_ident)), "take"] = set_number
                


df_filtered = df[(df['take'].notnull())]
df_filtered['fish_id'] = df_filtered['fish_id'].apply(lambda x: int(x)) 



df_filtered["angle"] = np.NAN


df_filtered["angle"] = df_filtered.apply(lambda x: (x.fish_head, x.tail_points[1]), axis=1)


df_filtered["rotation"] = np.NAN


import math

def calc_rotation(angle):
    x = angle[0][0] - angle[1][1]   #must invert, as squeleton points is y,x    
    y = angle[0][1] - angle[1][0]   #must invert, as squeleton points is y,x   
    dual_degree = math.atan2(y*-1, x) * 180 / np.pi
    return dual_degree

df_filtered["rotation"] = df_filtered["angle"].apply(calc_rotation)

df_filtered['abs_rotation'] = abs(df_filtered['rotation'])


df_filtered['diff'] = df_filtered.groupby(["quadrant", "fish_id", "take"])['abs_rotation'].diff() #abs(df_filtered['abs_rotation'] - df_filtered['abs_rotation'].shift(1))


df_filtered["displace"] = np.NAN

for quadrant in quadrante_numbers:
    for fish_ident in fish_identification:
        the_takes = df_filtered[(df_filtered["quadrant"] == quadrant) & (df_filtered["fish_id"] == fish_ident)]["take"].unique()
        for take_id in the_takes:
            list_frames = df_filtered[(df_filtered["quadrant"] == quadrant) & (df_filtered["fish_id"] == fish_ident) & (df_filtered["take"] == take_id)].index.values
            for idx, value in enumerate(list_frames):
                if idx > 1:                    
                               
                    the_value = df_filtered.loc[(df_filtered.index==value) & (df_filtered.quadrant == quadrant) & (df_filtered["take"] == take_id ) & (df_filtered.fish_id == fish_ident ), 'fish_head'].iloc[0]
                    previous_value = df_filtered.loc[(df_filtered.index==list_frames[idx-1]) & (df_filtered.quadrant == quadrant) & (df_filtered["take"] == take_id) & (df_filtered.fish_id == fish_ident ), 'fish_head'].iloc[0]           
                                        
                    df_filtered.loc[(df_filtered.index==value) & (df_filtered.quadrant == quadrant) & (df_filtered["take"] == take_id ) & (df_filtered.fish_id == fish_ident ), 'displace'] = ((the_value[0] - previous_value[0])**2 \
                    + (the_value[1] - previous_value[1])**2)**0.5
                            

df_filtered["next_predicted"] = None
from sklearn.linear_model import LinearRegression

def reg(row):
    coord=row["angle"]
    distance = row["displace"]
 
    
    if not np.isnan(distance):
        print("calculate")
        x = np.array([coord[1][1], coord[0][0]]).reshape((-1, 1)) # need to invert as tail is y and x and  not xy
        y = np.array([coord[1][0], coord[0][1]])
        model_regression = LinearRegression()
        model_regression.fit(x, y)
        model_regression = LinearRegression().fit(x, y)
        
        #calculate the next x based on distance of body and head
        calc = coord[1][1] - coord[0][0]
        if calc > 0:
            calc_x = coord[0][0] - distance
        else:
            calc_x = coord[0][0] + distance
            
        
        x = np.array([calc_x]).reshape((-1, 1))
        y_pred = model_regression.predict(x)
        return (int(calc_x), int(y_pred[0]))
    else:
        return None

df_filtered["next_predicted"] = df_filtered.apply(reg, axis = 1)

df_filtered['diff_pred'] = np.NaN

for quadrant in quadrante_numbers:
    for fish_ident in fish_identification:
        the_takes = df_filtered[(df_filtered["quadrant"] == quadrant) & (df_filtered["fish_id"] == fish_ident)]["take"].unique()
        for take_id in the_takes:
            list_frames = df_filtered[(df_filtered["quadrant"] == quadrant) & (df_filtered["fish_id"] == fish_ident) & (df_filtered["take"] == take_id)].index.values
            #print(list_frames)
            for idx, value in enumerate(list_frames):
                if idx > 0:      
                                                                        
                    current_position = df_filtered.loc[(df_filtered.index==value) & (df_filtered.quadrant == quadrant) & (df_filtered["take"] == take_id ) & (df_filtered.fish_id == fish_ident ), 'angle'].iloc[0][0]
                    predicted_position = df_filtered.loc[(df_filtered.index==list_frames[idx-1]) & (df_filtered.quadrant == quadrant) & (df_filtered["take"] == take_id) & (df_filtered.fish_id == fish_ident ), 'next_predicted'].iloc[0]           
                    if isinstance(predicted_position, tuple):
                        distance = ((current_position[0] - predicted_position[0])**2 + (current_position[1] - predicted_position[1])**2)**0.5
                        df_filtered.loc[(df_filtered.index==value) & (df_filtered.quadrant == quadrant) & (df_filtered["take"] == take_id ) & (df_filtered.fish_id == fish_ident ), 'diff_pred'] = distance
                                
                    

df_filtered['sequence'] = None
sequence_number = 1
for quadrant in quadrante_numbers:
    for fish_ident in fish_identification:
        the_takes = df_filtered[(df_filtered["quadrant"] == quadrant) & (df_filtered["fish_id"] == fish_ident)]["take"].unique()
        for take_id in the_takes:
            list_frames = df_filtered[(df_filtered["quadrant"] == quadrant) & (df_filtered["fish_id"] == fish_ident) & (df_filtered["take"] == take_id)].index.values
            for idx, value in enumerate(list_frames):                
                the_fish = df_filtered[(df_filtered.index == value) & (df_filtered["quadrant"] == quadrant) & (df_filtered["fish_id"] == fish_ident) & (df_filtered["take"] == take_id)]
                
                if  (the_fish['diff_pred'].iloc[0] < 3) and (the_fish['diff'].iloc[0] < 2)  and (the_fish['displace'].iloc[0] < 6.5):
                                      
                    df_filtered.loc[(df_filtered.index == value) & (df_filtered["quadrant"] == quadrant) & (df_filtered["fish_id"] == fish_ident) & (df_filtered["take"] == take_id), 'sequence'] = sequence_number 
                else:                   
                    sequence_number = sequence_number + 1
                   



count_df = df_filtered.groupby('sequence')['sequence'].count()



filtered_count = count_df[count_df > 5]


filtered_count = filtered_count.index.tolist()


filtered_count = set(filtered_count)


final_df = df_filtered[df_filtered['sequence'].isin(filtered_count)]


final_df["coord_plus_seq"] = np.NaN
final_df["coord_plus_seq"] = final_df.apply(lambda x: x.fish_head + (int(x.sequence),) + ((x.tail_points),), axis = 1)



final_df = final_df[['length_of_fish', 'center_of_mass', 'fish_tail',
       'fish_head', 'quadrant', 'fish_area', 'fish_id', 'tail_points', 
       'quad_coord', 'sequence', "take"]]

final_df.to_csv(path_to_save + '/rotated_' + expe + '.csv', mode='w', index=True, header=True)
