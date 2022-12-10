import numpy as np
import pandas as pd
import math
import sys
import ast
import os
import pathlib
import math
from matplotlib import pyplot as plt


path_to_csv = "C:/Users/marcio/Videos/Ian_videos/filtered_20191113_1302_53-2_L_A.csv" #sys.argv[1]
path_to_save = "C:/Users/marcio/Videos/Ian_videos/croped_Ian/errors" #sys.argv[2]

final_path = pathlib.PurePath(path_to_csv)
expe = final_path.name[:-4]

if os.path.exists(path_to_save + "/wave_corrected_" + final_path.name):
    print("CSV resulting file exist, remove it first")
    quit()
else:
    print("CSV resulting file does not exist, it will be created")

df = pd.read_csv(path_to_csv).set_index('frame_number')

df["tail_poly_corrected"] = None
df["distances"] = np.NaN

df['fish_head'] = df['fish_head'].apply(ast.literal_eval)
df['center_of_mass'] = df['center_of_mass'].apply(ast.literal_eval)
df['tail_points'] = df['tail_points'].apply(ast.literal_eval)
df["angle_corr_tail"] = df['tail_points']



for quadrant in [0,1,2,3]: # [0,1,2,3]
    for fish_ident in [1,2]: # [1,2]    
        frames_numbers = df[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident)].index.values   
        for idx_frame in frames_numbers:
            if idx_frame%100==0: 
                print(quadrant, fish_ident, idx_frame)            
            the_row = df.loc[(df.index == idx_frame) & (df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident)]
            #print(the_row)
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
            
            df.loc[(df.index == idx_frame) & (df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident), 'angle_corr_tail'] = str(corrected_tails_points)
            #print(df.loc[(df.index == idx_frame) & (df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident), 'angle_corr_tail'])
            pass

        def distances(t):
            t = ast.literal_eval(t)
            distances = []
            for idx, _ in enumerate(t):
                if idx > 0:
                    distance = math.hypot(t[idx][0] - t[idx-1][0], t[idx][1] - t[idx-1][1])
                    distances.append(distance)
            return sum(distances)
        
        #def literal_set(list_):
            #if isinstance(list_, str):
                #return ast.literal_eval(list_)
            
            
        #df['angle_corr_tail'] = df['angle_corr_tail'].apply(literal_set)

        df.loc[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident), 'distances'] = df.loc[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident), 'angle_corr_tail'].apply(distances)

        standart_distance = df.loc[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident), 'distances'].median()

        def coord_calc(row):
            print("the frame")
            print(row.name)
            tail_points_str = row["angle_corr_tail"]
            tail_points = ast.literal_eval(tail_points_str)
            x_tail_to_check = list(zip(*tail_points))[0]
            if sorted(x_tail_to_check) != x_tail_to_check:
                return
            summed_distances = []
            for idx, _ in enumerate(tail_points):
                if idx > 0:
                    distance = math.hypot(tail_points[idx][0] - tail_points[idx-1][0], tail_points[idx][1] - tail_points[idx-1][1])          
                    summed_distances.append(distance)
            summed_distances = sum(summed_distances) 
                    
            if summed_distances > standart_distance:
                while summed_distances > standart_distance:
                    print(row.name)

                    for i in range(1, 5):
                        tail_points[i][0] = ((tail_points[i][0])*0.99) 
                    summed_distances_ind = []
                    for idx, _ in enumerate(tail_points):
                        if idx > 0:
                            distance = math.hypot(tail_points[idx][0] - tail_points[idx-1][0], tail_points[idx][1] - tail_points[idx-1][1])          
                            summed_distances_ind.append(distance)
                    summed_distances = sum(summed_distances_ind)
                    print(summed_distances)
            else:
                while summed_distances < standart_distance:
                    print(row.name)

                    for i in range(1, 5):
                        tail_points[i][0] = ((tail_points[i][0])*1.01)           
                    summed_distances_ind = []
                    for idx, _ in enumerate(tail_points):
                        if idx > 0:
                            distance = math.hypot(tail_points[idx][0] - tail_points[idx-1][0], tail_points[idx][1] - tail_points[idx-1][1])          
                            summed_distances_ind.append(distance)
                    summed_distances = sum(summed_distances_ind)
                    print(summed_distances)

            x_tail = list(zip(*tail_points))[0]
            y_tail = list(zip(*tail_points))[1]  
          
            
            #xnew = np.linspace(x_tail[0], x_tail[-1])            
            model3 = np.poly1d(np.polyfit(x_tail, y_tail, 3))            
            y_modeled = tuple(model3(x_tail))
            x_modeled = tuple(x_tail)
            #plt.figure(1)         
            #plt.plot(x_modeled, model3(x_modeled)) 
            #plt.pause(1)
            
            
            
            df.loc[(df.index == row.name) & (df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident), 'angle_corr_tail'] = str(tuple(zip(x_modeled, y_modeled)))
            

        df.loc[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident)].apply(coord_calc, axis=1) 
        


df = df[['length_of_fish', 'center_of_mass', 'fish_tail', 'fish_head', 'quadrant', 'fish_area', 
         'fish_id', 'quad_coord', 'sequence','angle_corr_tail', 'take', "sum_chanel_B", "sum_chanel_G", 
         'sum_chanel_R', "avg_chanel_B", "avg_chanel_G", "avg_chanel_R", "count_chanel"]]

df.to_csv(path_to_save + '/wave_corrected_' + final_path.name[8:], mode='w', index=True, header=True)





