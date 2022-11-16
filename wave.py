

# Check if camera opened successfully
import pickle
import winsound

import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
import math

df = pd.read_csv("C:/Users/marcio/Documents/fish_analyzer_rotated.csv").set_index('frame_number')

#tail_coords



#df.columns=["fish_head_x", "fish_head_y", "sequence", "tail_coords"]
df["angle_corr_tail"] = None
df["tail_poly_corrected"] = None
df["distances"] = np.NaN
df["tail_uniformity"] = np.NaN
#df["good_tail"] = False
#df["tail_coords"] = None





plt.ion()


cap = cv2.VideoCapture('C:/Users/marcio/Videos/Ian_videos/20191121_1454_iCab_L_C.avi')
final_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
import ast
df['fish_head'] = df['fish_head'].apply(ast.literal_eval)
df['center_of_mass'] = df['center_of_mass'].apply(ast.literal_eval)
df['tail_points'] = df['tail_points'].apply(ast.literal_eval)



if (cap.isOpened()== False): 
  print("Error opening video stream or file")

from itertools import cycle

color_cycle = cycle(((0,0,255),(0,255,0),(255,0,0)))

#frame_n = 0


for quadrant in [1]:
  for fish_ident in [1]:     
    frames_numbers = df[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident)].index.values   

    previous_id = 0
    # Read until video is completed
    for idx_frame in range(0,final_frame,1):  
      if idx_frame in frames_numbers:  
        print(idx_frame)

          
        
        cap.set(1, idx_frame)
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
          
              the_row = df.loc[(df.index == idx_frame) & (df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident)]
              #print(frames_numbers[idx_frame])
              if previous_id != the_row["sequence"].iloc[0]:
              #if previous_id != frames_numbers[idx_frame][2]:
                color = next(color_cycle)
              cv2.rectangle(frame, (the_row["center_of_mass"].iloc[0][0]-30,the_row["center_of_mass"].iloc[0][1]-30), (the_row["center_of_mass"].iloc[0][0]+30, the_row["center_of_mass"].iloc[0][1]+30), color, 2)
              font = cv2.FONT_HERSHEY_SIMPLEX
              #cv2.putText(frame, str(the_row["fish_head"]iloc[0][0], the_row["fish_head"].iloc[0][0]-30), (the_row["fish_head"].iloc[0][1]-30, the_row["fish_head"].iloc[0][1]-30), font, 0.5, (0,0,0), 1) 
              cv2.putText(frame, str(the_row["sequence"].iloc[0]),(the_row["center_of_mass"].iloc[0][0]-30,the_row["center_of_mass"].iloc[0][1]-30), font, 0.5, (0,0,0), 1)
              previous_id = the_row["sequence"].iloc[0]
              
              
              for coords in the_row["tail_points"].iloc[0]:      
              #tail plot
              #cv2.circle(drawn_image, fish_tail_local[c], 2, (0, 0, 255), -1)
                cv2.circle(frame, (coords[1], coords[0]), 2, (0, 0, 255), -1)
                
        

              
              
              cv2.imshow('Frame', frame)
              
              #cv2.waitKey(300)
              
      

              if 1 == 1:
                # Define x, y, and xnew to resample at.      
                x_list = list(zip(*the_row["tail_points"].iloc[0]))[1]
                #x_final = [float(i)/min(x_list) for i in x_list]


                
                #print(y)
                y_list_inv = list(zip(*the_row["tail_points"].iloc[0]))[0]
                y_list = [i for i in y_list_inv] # invert the y axis 870
                
                #y_final = [float(i)/min(y_list) for i in y_list]
                
                
                tuple_y_inv = tuple(zip(x_list, y_list))
                
              
                import math

                def calc_rotation(coords_body):
                  x = coords_body[1][0] - coords_body[2][0]             
                  y = coords_body[1][1] - coords_body[2][1]              
                  dual_degree = math.atan2(y*-1, x) * 180 / np.pi
                  if dual_degree < 0:
                    #pass
                    dual_degree = 360+dual_degree                              
                  return dual_degree

                angle = calc_rotation(tuple_y_inv)

                
                
                import math

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
                
          
        
            
                

                x_list_2 = list(zip(*corrected_angle))[0]
                y_list_2 = list(zip(*corrected_angle))[1]
                
                x_norm = x_list_2 #[i-x_list_2[0] for i in x_list_2]
                y_norm = y_list_2 #[i-y_list_2[0] for i in y_list_2]
                
                
                
                x_norm_2 = [float(i)/x_norm[0] for i in x_norm]
                y_norm_2 = [float(i)/y_norm[1] for i in y_norm]
                
                
                
                corrected_tails_points = [list(a) for a in zip(x_norm_2, y_norm_2)]
                
                #insert normalized to df
                df.loc[(df.index == idx_frame) & (df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident), 'angle_corr_tail'] = str(corrected_tails_points)
            
               
                pass
        
             
              else:
                pass
                        
              
              
          
              
              
              
            
            
            
            
            

              #cv2.waitKey(100)
              #cv2.destroyAllWindows()          
              #frame_n +=1
        
              
              
              
          # Press Q on keyboard to exit
              if cv2.waitKey(25) & 0xFF == ord('q'):
        
                  break
        else:
          print("no frames 1")
          break
          
          
    # Break the loop if no frame is retrieved
      else:
        #print("no frames 2")
        continue
    #cv2.waitKey(0)  
    cv2.destroyAllWindows() 



    

    def distances(t):
      #x_tuple = tuple(zip(*x))[0]
      #y_tuple = tuple(zip(*x))[1]
      distances = []
      for idx, _ in enumerate(t):
        if idx > 0:
          distance = math.hypot(t[idx][0] - t[idx-1][0], t[idx][1] - t[idx-1][1])
          distances.append(distance)
          
      return sum(distances)
    
    #df['angle_corr_tail'] = df['angle_corr_tail'].apply(ast.literal_eval)
    def literal_set(list_):
      if isinstance(list_, str):
        return ast.literal_eval(list_)
    df['angle_corr_tail'] = df['angle_corr_tail'].apply(literal_set)


    df.loc[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident), 'distances'] = df.loc[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident), 'angle_corr_tail'].apply(distances)
    #df["distances"] = df["angle_corr_tail"].apply(distances)

    #df["max_row_distance"] = np.NaN

    standart_distance = df.loc[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident), 'distances'].max()
    #summed_distances = df["distances"].mean()/4

    def coord_calc(row):
      tail_points = row["angle_corr_tail"] 
     
      summed_distances = []
      for idx, _ in enumerate(tail_points):
        if idx > 0:
          distance = math.hypot(tail_points[idx][0] - tail_points[idx-1][0], tail_points[idx][1] - tail_points[idx-1][1])          
          summed_distances.append(distance)
      summed_distances = sum(summed_distances) 
      print(summed_distances)
      print(standart_distance)
         
      if summed_distances > standart_distance:
        #print(tail_points)
        while summed_distances > standart_distance:
          #slope, intercept, r, p, std_err = stats.linregress([tail_points[idx][0], tail_points[idx-1][0]], [tail_points[idx][1], tail_points[idx-1][1]])
          for i in range(1, 5):
            tail_points[i][0] = ((tail_points[i][0] - 1)*0.99) + 1
            print("r1")        
          summed_distances = []
          for idx, _ in enumerate(tail_points):
            if idx > 0:
              distance = math.hypot(tail_points[idx][0] - tail_points[idx-1][0], tail_points[idx][1] - tail_points[idx-1][1])          
              summed_distances.append(distance)
          summed_distances = sum(summed_distances) 
      else:
        print(tail_points)
        while summed_distances < standart_distance:
          for i in range(1, 5):
            tail_points[i][0] = ((tail_points[i][0] - 1)*1.01) + 1          
          summed_distances = []
          for idx, _ in enumerate(tail_points):
            if idx > 0:
              distance = math.hypot(tail_points[idx][0] - tail_points[idx-1][0], tail_points[idx][1] - tail_points[idx-1][1])          
              summed_distances.append(distance)
          summed_distances = sum(summed_distances)
      
      print(tail_points) 

      




    df.loc[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident)].apply(coord_calc, axis=1) 



    
    def vari(the_list):
      if isinstance(the_list, list):
        y_list = list(zip(*the_list))[1]
        cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100        
        result = cv(y_list)
        if result is not None:      
          return result
        else:
          return np.NaN
      else:
        np.NaN
        
    df["tail_uniformity"] = df. angle_corr_tail.apply(vari)
    
    #df["good_tail"] = df["tail_uniformity"].apply(lambda x: True if (x < 0.005) else False)
    #plt.hist(df["tail_uniformity"], bins = 30)
    #plt.show()

    sequences = df.loc[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident), 'sequence'].unique()

    #sequences = df.loc[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident), "sequence"].unique()

    # Read until video is completed
    #sequences = df["sequence"].unique()
    for sequence in sequences:
      #sequence = 110
      
      sub_seq = df.loc[df["sequence"]== sequence]
      for idx_frame in sub_seq.index: 
      
      
        if idx_frame in frames_numbers:  
          print(idx_frame)

            
          
          cap.set(1, idx_frame)
          
          # Capture frame-by-frame
          ret, frame = cap.read()
          if ret == True:
          
            the_row = df.loc[(df.index == idx_frame) & (df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident)]
            #print(frames_numbers[idx_frame])
            if previous_id != the_row["sequence"].iloc[0]:
            #if previous_id != frames_numbers[idx_frame][2]:
              color = next(color_cycle)
            cv2.rectangle(frame, (the_row["center_of_mass"].iloc[0][0]-30,the_row["center_of_mass"].iloc[0][1]-30), (the_row["center_of_mass"].iloc[0][0]+30, the_row["center_of_mass"].iloc[0][1]+30), color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(frame, str(the_row["fish_head"]iloc[0][0], the_row["fish_head"].iloc[0][0]-30), (the_row["fish_head"].iloc[0][1]-30, the_row["fish_head"].iloc[0][1]-30), font, 0.5, (0,0,0), 1) 
            cv2.putText(frame, str(the_row["sequence"].iloc[0]) + " " + str(the_row["tail_uniformity"].iloc[0]),(the_row["center_of_mass"].iloc[0][0]-30,the_row["center_of_mass"].iloc[0][1]-30), font, 0.5, (0,0,0), 1)
            previous_id = the_row["sequence"].iloc[0]
            
            
            for coords in the_row["tail_points"].iloc[0]:      
            #tail plot
            #cv2.circle(drawn_image, fish_tail_local[c], 2, (0, 0, 255), -1)
              cv2.circle(frame, (coords[1], coords[0]), 2, (0, 0, 255), -1)
              
        

              
              
                
                
            tail_coordinates = the_row['angle_corr_tail'].iloc[0]
            x_list = list(zip(*tail_coordinates))[0]
            y_list = list(zip(*tail_coordinates))[1]
            
            print(x_list)
            
            
            
            xnew = np.linspace(x_list[0], x_list[-1])           
              
            #f_cubic = interp1d(x_list, y_list, kind='quadratic')
            #plt.plot(xnew, f_cubic(xnew), label='quadratic')
            
            print(the_row["tail_uniformity"])
            #if the_row["good_tail"].iloc[0] == True:
            plt.figure(1)
            #plt.ylim(0.5, 1.5)
            #plt.xlim(1, 2)
            #plt.xlim(1, 1.08)
        
          
            plt.plot(x_list, y_list, 'o', label='data')          
            model4 = np.poly1d(np.polyfit(x_list, y_list, 3))
            plt.plot(x_list, y_list)
            #plt.show()
            #plt.pause(0.3)
                  
            
            plt.figure(2)
            #plt.ylim(0.5, 1.5)
            #plt.xlim(1, 2)
            #plt.xlim(0.970, 1.08)
            plt.plot(x_list, y_list, 'o', label='data')          
            #model4 = np.poly1d(np.polyfit(x_list, y_list, 3))
            
            
            #substitue values in coords by the poly fir found
            
            y_modeled = tuple(model4(x_list))
            x_modeled = tuple(x_list)
            y_modulated = [(n*1) for n in y_modeled]
            x_modulated = [(n*1) for n in x_modeled]
            
            final_tails = str(tuple(zip(x_modulated, y_modulated)))
            
            df.loc[(df.index == idx_frame) & (df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident), "tail_coords"] = final_tails

            print(final_tails)
            #x_norm = [float(i)/x_modeled[0] for i in x_modeled]
            #y_norm = [float(i)/y_modeled[1] for i in y_modeled]
            
            #df.iloc[idx_frame, 5] = 
            
            #df.at[idx_frame, "tail_poly_corrected"] = tuple(zip(x_norm, y_norm))
            
            
            
            
            #.at[idx_frame] = tuple(zip(x_modeled, y_modeled))
            
            #tuple(zip(x_modeled, model4(x_modeled)))

            
            
            
            
            
            
            plt.plot(xnew, model4(xnew))
            #plt.show()
            plt.pause(1)  
            #cv2.waitKey(3)       
            
            
            #plt.pause(2)
            #plt.clf()
                      
            

           
            
            cv2.imshow('Frame', frame)
            #cv2.waitKey(1)
            
              
            #plt.close(1)
            
            
                
                
                  
                
                
                
                          
      
        
      # Break the loop if no frame is retrieved
      else:
        #print("no frames 2")
        continue
      #cv2.waitKey(0)

    plt.clf()  
    cv2.destroyAllWindows()
#df_to_analyze = df[["sequence", "tail_poly_corrected"]]
#df_to_analyze["head"] = df.apply(lambda x: (x.fish_head_x, x.fish_head_y), axis=1)

#df_to_analyze.rename(columns={"angle_corr_tail":"tail_coords"}, inplace=True)
#df_to_analyze.rename({'tail_poly_corrected': 'tail_coords'}, axis=1, inplace=True)

#df.loc[:, 'points'] = df.points.apply(lambda x: x*2)
#df['tail_coords'] = None
#df['tail_coords'] = df["tail_poly_corrected"].apply(lambda x: tuple(tuple([int((n*10000)-10000) for n in sub]) for sub in x))
#df_to_analyze.loc[:, 'tail_coords'] = df_to_analyze.tail_coords.apply(lambda x: tuple(tuple([(int(n*1000)) for n in sub]) for sub in x))

print(df.columns)
df = df[['length_of_fish', 'center_of_mass', 'fish_tail', 'fish_head', 'quadrant', 'fish_area', 'fish_id', 'quad_coord', 'sequence', 'tail_uniformity','tail_coords', 'take']]
df.to_csv('C:/Users/marcio/Documents/fish_analyzer_final' + '.csv', mode='w', index=True, header=True)








