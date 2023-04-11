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

df = pd.read_csv("C:/Users/marcio/Videos/Ian_videos/croped_Ian/errors/wave_corrected__20191118_1021_7-2_L_B.csv").set_index('frame_number')

#tail_coords



#df.columns=["fish_head_x", "fish_head_y", "sequence", "tail_coords"]
#df["angle_corr_tail"] = None
#df["tail_poly_corrected"] = None
#df["distances"] = np.NaN
#df["good_tail"] = False
#df["tail_coords"] = None





#################plt.ion()


cap = cv2.VideoCapture('C:/Users/marcio/Videos/Ian_videos/20191114_1023_7-2_R_A.avi')
final_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
import ast
df['fish_head'] = df['fish_head'].apply(ast.literal_eval)
df['center_of_mass'] = df['center_of_mass'].apply(ast.literal_eval)
df['angle_corr_tail'] = df['angle_corr_tail'].apply(ast.literal_eval)



if (cap.isOpened()== False): 
  print("Error opening video stream or file")

from itertools import cycle

color_cycle = cycle(((0,0,255),(0,255,0),(255,0,0)))
color = None
#frame_n = 0


for quadrant in [0]: # [0,1,2,3]
  for fish_ident in [1]: # [1,2]    
    frames_numbers = df[(df["quadrant"] == quadrant) & (df["fish_id"] == fish_ident)].index.values   
    print(frames_numbers)
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
              
              
              for coords in the_row["angle_corr_tail"].iloc[0]:      
              #tail plot
              #cv2.circle(drawn_image, fish_tail_local[c], 2, (0, 0, 255), -1)
                #cv2.circle(frame, (coords[1], coords[0]), 2, (0, 0, 255), -1)
                pass
        

              
              
              cv2.imshow('Frame', frame)
              
              #cv2.waitKey(300)
              
      

             
              
              
              
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



    

    
            
            
    

           
     