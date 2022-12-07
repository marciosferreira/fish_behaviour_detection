import numpy as np
import pandas as pd
import math
import sys
import ast
import os
import pathlib
import math

path_to_csv = "C:/Users/marcio/Videos/Ian_videos/croped_Ian/errors/rotated_20191111_1527_5-1_L_A.csv" #sys.argv[1]
path_to_save = "C:/Users/marcio/Videos/Ian_videos/croped_Ian/errors" #sys.argv[2]

final_path = pathlib.PurePath(path_to_csv)
expe = final_path.name[:-4]

if os.path.exists(path_to_save + "/wave_corrected_" + final_path.name):
    print("CSV resulting file exist, remove it first")
    quit()
else:
    print("CSV resulting file does not exist, it will be created")

df = pd.read_csv(path_to_csv).set_index('frame_number')

df["angle_corr_tail"] = None
df["tail_poly_corrected"] = None
df["distances"] = np.NaN

df['fish_head'] = df['fish_head'].apply(ast.literal_eval)
df['center_of_mass'] = df['center_of_mass'].apply(ast.literal_eval)
df['tail_points'] = df['tail_points'].apply(ast.literal_eval)

for quadrant in [0,1,2,3]: # [0,1,2,3]
    for fish_ident in [1,2]: # [1,2]    
        frames_numbers = df[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident)].index.values   
        for idx_frame in frames_numbers:            
            the_row = df.loc[(df.index == idx_frame) & (df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident)]
            # Define x, y, and xnew to resample at.      
            x_list = list(zip(*the_row["tail_points"].iloc[0]))[1]           
            y_list_inv = list(zip(*the_row["tail_points"].iloc[0]))[0]
            y_list = [i for i in y_list_inv] # invert the y axis 870            
            tuple_y_inv = tuple(zip(x_list, y_list))

            def calc_rotation(coords_body):
                x = coords_body[1][0] - coords_body[2][0]             
                y = coords_body[1][1] - coords_body[2][1]              
                dual_degree = math.atan2(y*-1, x) * 180 / np.pi
                if dual_degree < 0:
                    #pass
                    dual_degree = 360+dual_degree                              
                return dual_degree

            angle = calc_rotation(tuple_y_inv)           

            def rotate(points, radians, origin):
                x, y = points
                ox, oy = origin

                qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
                qy = oy + -math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)
                
                return qx, qy
            
            
            if angle > 180:
                angle = 360-angle+180
            else:
                angle = 180-angle
                

            corrected_angle = [rotate(i, math.radians(angle), tuple_y_inv[1]) for i in tuple_y_inv]
            
            
            x_norm = list(zip(*corrected_angle))[0]
            y_norm = list(zip(*corrected_angle))[1]            
            
            x_norm_2 = [float(i) - x_norm[0] for i in x_norm]
            y_norm_2 = [float(i) - y_norm[1] for i in y_norm]  
            
            corrected_tails_points = [list(a) for a in zip(x_norm_2, y_norm_2)]
            
            #insert modeled to df
            df.loc[(df.index == idx_frame) & (df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident), 'angle_corr_tail'] = str(corrected_tails_points)
        

        def distances(t):
            distances = []
            for idx, _ in enumerate(t):
                if idx > 0:
                    distance = math.hypot(t[idx][0] - t[idx-1][0], t[idx][1] - t[idx-1][1])
                    distances.append(distance)
            return sum(distances)
        
        def literal_set(list_):
            if isinstance(list_, str):
                return ast.literal_eval(list_)
            
            
        df['angle_corr_tail'] = df['angle_corr_tail'].apply(literal_set)

        df.loc[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident), 'distances'] = df.loc[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident), 'angle_corr_tail'].apply(distances)

        standart_distance = df.loc[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident), 'distances'].mean()

        def coord_calc(row):
            tail_points = row["angle_corr_tail"]
            summed_distances = []
            for idx, _ in enumerate(tail_points):
                if idx > 0:
                    distance = math.hypot(tail_points[idx][0] - tail_points[idx-1][0], tail_points[idx][1] - tail_points[idx-1][1])          
                    summed_distances.append(distance)
            summed_distances = sum(summed_distances)         
            if summed_distances > standart_distance:
                while summed_distances > standart_distance:
                    for i in range(1, 5):
                        tail_points[i][0] = ((tail_points[i][0])*0.99) 
                    summed_distances = []
                    for idx, _ in enumerate(tail_points):
                        if idx > 0:
                            distance = math.hypot(tail_points[idx][0] - tail_points[idx-1][0], tail_points[idx][1] - tail_points[idx-1][1])          
                            summed_distances.append(distance)
                    summed_distances = sum(summed_distances) 
            else:
                while summed_distances < standart_distance:
                    for i in range(1, 5):
                        tail_points[i][0] = ((tail_points[i][0])*1.01)           
                    summed_distances = []
                    for idx, _ in enumerate(tail_points):
                        if idx > 0:
                            distance = math.hypot(tail_points[idx][0] - tail_points[idx-1][0], tail_points[idx][1] - tail_points[idx-1][1])          
                            summed_distances.append(distance)
                    summed_distances = sum(summed_distances)
        

        df.loc[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident)].apply(coord_calc, axis=1) 
        
        def modeled(tail_points):
            x_tail = list(zip(*tail_points))[0]
            y_tail = list(zip(*tail_points))[1]  
            model4 = np.poly1d(np.polyfit(x_tail, y_tail, 3))                    
            y_modeled = tuple(model4(x_tail))
            x_modeled = tuple(y_tail)
            return [list(a) for a in zip(x_modeled, y_modeled)]

        
        df.loc[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident), 'angle_corr_tail'].apply(modeled)
        

df = df[['length_of_fish', 'center_of_mass', 'fish_tail', 'fish_head', 'quadrant', 'fish_area', 'fish_id', 'quad_coord', 'sequence','angle_corr_tail', 'take']]
df.to_csv(path_to_save + '/wave_corrected_' + final_path.name[8:], mode='w', index=True, header=True)





