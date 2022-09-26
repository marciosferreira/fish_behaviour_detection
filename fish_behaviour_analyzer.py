#Setup paramters
quadr = (0,1,2,3)     # 0 for left up, 1 for left down, 2 for right up, 3 for right down or None for the four. 
initial_frame = 1746  # will throw an error if not 2 fish are in each quadrant in the first frame (if any fish are overlapped).
final_frame = 1000000  # set a very big number to go until the end of the video. e.g: 1000000
path_to_video = 'C:/Users/marcio/Videos/Ian_videos/20191121_1454_iCab_L_C.avi'  # video path
background_img = 'C:/Users/marcio/Documents/background_1.jpg' #background image
path_to_save = "C:/Users/marcio/Documents/results_Ian"  #where to save results

#####################################################################################
import os
import pathlib

final_path = pathlib.PurePath(path_to_video)
file_name = final_path.name

if os.path.exists(path_to_save + "/" + file_name + '.csv'):
  os.remove(path_to_save + "/" + file_name + '.csv')
  print("CSV file exist, It has been removed to a new one be created")
else:
  print("CSV file does not exist, it will be created")

print(path_to_save + "/" + file_name + '.csv')
 
with open(path_to_save + "/" + file_name + '.csv', 'w') as fd:
  fd.write('frame_number, length_of_fish, center_of_mass, fish_tail, fish_head, quadrant, fish_area, fish_id\n')

from collections import deque
import cv2
#from cmath import nan
import scipy.stats as stats
import scikit_posthocs as sp
#from scipy.stats import skewgit branch
from scipy.stats import kurtosis
#from scipy.stats import iqr
#import statistics
#import seaborn as sns
#from cv2 import waitKey 
import numpy as np
import math
import pandas as pd
#import imagehash
#from PIL import Image     
#import time
#update_counter = 23
#import matplotlib.pyplot as plt 
#import imutils
fish_0_history = []
fish_1_history = []

make_copy_last_seen = {}
make_copy_last_seen[0] = True
make_copy_last_seen[1] = True
make_copy_last_seen[2] = True
make_copy_last_seen[3] = True




first = True
#from statsmodels.stats.multicomp import pairwise_tukeyhsd

calculo_area = []
went_ok = 0

active = {}
active[0] = "XY"
active[1] = "XY"
active[2] = "XY"
active[3] = "XY"

is_first_iteraction = {}
is_first_iteraction[0] = False
is_first_iteraction[1] = False
is_first_iteraction[2] = False
is_first_iteraction[3] = False

 
pd.set_option('display.max_columns', None) 
background_img = cv2.imread(background_img)
bw_back = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
bw_back = cv2.GaussianBlur(bw_back, (9,9) ,0) 

#create a blank image to plot everything on
blank_image = np.zeros((bw_back.shape[0], bw_back.shape[1], 3), np.uint8)

previous_df = None
lopp = 0

previous_id_fish_local = []

histograms_ids = {'10':deque(maxlen=60), '11':deque(maxlen=60), '12':deque(maxlen=60), '13':deque(maxlen=60), '20':deque(maxlen=60), '21':deque(maxlen=60), '22':deque(maxlen=60), '23':deque(maxlen=60)}
histograms_X_Y = {"X0":deque(maxlen=60), "X1":deque(maxlen=60), "X2":deque(maxlen=60), "X3":deque(maxlen=60), "Y0":deque(maxlen=60), "Y1":deque(maxlen=60), "Y2":deque(maxlen=60), "Y3":deque(maxlen=60)}

#cv2.imshow('background' , bw_back)

# Check if camera opened successfully
cap = cv2.VideoCapture(path_to_video)
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
for idx_frame in range(initial_frame,final_frame,1):   #3000 to 4000
  print(idx_frame)
    
  
  cap.set(1, idx_frame)
  
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    
    #########################################################################
    #detec quadrantes on first image
    # Load image, grayscale, median blur, sharpen image
    if idx_frame == initial_frame:
      #image = cv2.imread('1.pn)
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
      blur = cv2.medianBlur(gray, 9)
      sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
      sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

      # Threshold and morph close
      #thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
      thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
      kernel = np.ones((15,15),np.uint8)
      opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

      # Find contours and filter using threshold area
      cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cnts = cnts[0] if len(cnts) == 2 else cnts[1]

      max_area = (frame.shape[0] * frame.shape[1])/4
      min_area = max_area * 0.5
      quadrants_lines = []
      for c in cnts:
          area = cv2.contourArea(c)          
          if area > min_area and area < max_area:
              x,y,w,h = cv2.boundingRect(c)
              quadrants_lines.append((x,y,w,h))
                           
      if len(quadrants_lines) != 4:
        print("Number od detected quadrants is not 4")
        quit()          
    ###########################################################
    
    first_fish = []
    second_fish = []
    
    
    #x_y_line = (((quadrants_lines[1][0]+quadrants_lines[1][2]) - (quadrants_lines[0][0]))/2, ((quadrants_lines[3][1]) + (quadrants_lines[3][3])))
    #cv2.rectangle(frame, (int(x_y_line[0]), int(x_y_line[1])), (int(x_y_line[0])+100, int(x_y_line[1])+100), (36,255,12), 2)


    #imzz = cv2.resize(frame, (780, 780))              # Resize image   

    #cv2.imshow('Main',imzz)
    #cv2.waitKey(1)
     
    #########################################################################
    #block2
    # Here the script will extract information from each contour
    
    original_img = frame.copy()

    # Display the resulting frame
    bw_mainImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bw_mainImage = cv2.GaussianBlur(bw_mainImage, (9,9) ,0)

    diff = cv2.absdiff(bw_back, bw_mainImage)
          
    ret,thresh = cv2.threshold(diff,15,255,cv2.THRESH_BINARY)  
    

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    

    lenght_of_fish_local = []
    position_fish_local = []
    fish_tail_local = []
    fish_head_local = []
    idx_local = [] 
    id_fish_local = []
    quadrant_local = []
    fish_area = []
    #fish_color = []
    histograms = []
    countours_idx = [] # for use with template matching
    template_area = []
    template_blur = []
    template_dark = []
    counter = 0
    
    for idx, cnt in enumerate(contours):
                  
      area = cv2.contourArea(cnt) 
      
      #calculate aspect ratio for template quality filtering   

      if area > 200 and area < 1500:        
              
        #will be used to predict the size of the fish excluding the tail part
        fish_total_pixels = len(cnt)

        area = cv2.contourArea(cnt)

        #to define in which quadrant the fish belongs to
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"]) 

        # determine the most extreme points along the contour
        extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
        extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
        extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
        extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
        list_of_points = [extLeft, extRight, extTop, extBot]

        #calculate the tail by discovering the fartherst point from center of mass of fish
        distances_tail = []  
        for l in list_of_points:                     
          distance = math.sqrt(   (cX-l[0])**2 + (cY-l[1])**2    )
          distances_tail.append(distance)

        max_value = max(distances_tail)
        max_index_tail = distances_tail.index(max_value)

        #discovering the head by calculating the fartherst point from tail
        distances_head = []
        for l in cnt:
                  
          distance = math.sqrt(   (l[0][0]-list_of_points[max_index_tail][0])**2 + (l[0][1]-list_of_points[max_index_tail][1])**2    )
          distances_head.append(distance)

        max_value = max(distances_head)
        max_index_head = distances_head.index(max_value)

        arr = np.array(distances_head)
        arr = arr.argsort()[-10:][::-1]
        list_final = arr.tolist()
    

        farthest_values = []
        for d in list_final:
          list_value = cnt[d].tolist()              
          farthest_values.append(list_value)

        arr = np.array(farthest_values)
  
        
        aver_head = np.mean(arr, axis=0).astype(int)
        
        #the head coordinates
        aver_head = (aver_head[0][0], aver_head[0][1])


        #calculate the center of mass of the fish excluding the tail
        distances_cm = []
        for j in cnt:
                  
          #append distances betwen head and each point of the fish
          distance = math.sqrt(   (j[0][0]-int(aver_head[0]))**2 + (j[0][1]-int(aver_head[1]))**2    )
          distances_cm.append(distance)
          max_distance = max(distances_cm)            

        slowest_indices = sorted(range(len(distances_cm)), key = lambda sub: distances_cm[sub])[:int(fish_total_pixels)]
        #slowest_indices_for_template = sorted(range(len(distances_cm)), key = lambda sub: distances_cm[sub])[:int(fish_total_pixels)]                   
        
            
        nearests_values = []
        #nearest_values_for_template = []
        
        for e in slowest_indices:
          if distances_cm[e] < int(max_distance*0.7):
            list_value_cm = cnt[e].tolist()              
            nearests_values.append(list_value_cm)

        arr = np.array(nearests_values)
        
   
        area_rec = cv2.contourArea(arr)
       
        aver_cm = np.mean(arr, axis=0).astype(int)
        
        #fish_center of mass
        fish_COM = (aver_cm[0][0], aver_cm[0][1])  # x and y axis
        
        fish_pectoral_lenght = math.sqrt( (aver_head[0] - aver_cm[0][0])  **2 + (aver_head[1] - aver_cm[0][1])**2    )  
        
         # drawn quadrants
        for coordinates in quadrants_lines:
          x,y,w,h = coordinates
          cv2.rectangle(frame, (x, y), (x + w, y + h), (36,255,12), 2)
          cv2.imshow('Main',frame)
        
        if (fish_COM[0] < (quadrants_lines[3][0] + quadrants_lines[3][2] +10) and fish_COM[1] < ((quadrants_lines[3][1] + quadrants_lines[3][3] +10))):  
          quadrant_value = 0
        elif (fish_COM[0] < (quadrants_lines[0][0] + quadrants_lines[0][2] +10) and fish_COM[1] > ((quadrants_lines[0][1] -10))):
          quadrant_value = 1
        elif (fish_COM[0] > (quadrants_lines[2][0] -10) and fish_COM[1] < ((quadrants_lines[2][1] + quadrants_lines[2][3] +10))):
          quadrant_value = 2
        elif (fish_COM[0] > (quadrants_lines[1][0] -10) and fish_COM[1] > (quadrants_lines[1][1] -10)):
          quadrant_value = 3
        else:
          print("erro!!!!!!!!!!!")
          quit()           
      

        #store variables locally in the loop for immediate calculation purposes
        idx_local.append(counter)          
        lenght_of_fish_local.append(fish_pectoral_lenght)
        position_fish_local.append(fish_COM)
        fish_tail_local.append(list_of_points[max_index_tail])
        fish_head_local.append(aver_head)
        quadrant_local.append(quadrant_value)
        fish_area.append(area)        
        histograms.append(fish_pectoral_lenght)   # need to fix afterwards
        countours_idx.append(0)
        template_area.append(area_rec)
        template_blur.append(0)
        template_dark.append(0)        
                      
        counter +=1
    
    
    dframe = pd.DataFrame(idx_local)
    dframe['lenght_of_fish_local'] = lenght_of_fish_local
    dframe['position_fish_local'] = position_fish_local
    dframe['fish_tail_local'] = fish_tail_local
    dframe['fish_head_local'] = fish_head_local
    dframe['quadrant_local'] = quadrant_local
    dframe['fish_area'] = fish_area
    dframe['cnt_idx'] = countours_idx
    dframe["fish_id"] = np.nan
   
   
    #######################################################################################################
    
    #create a XY if is the very begining of the script
    
    """if previous_df is None: 
           
      active[0] = "XY"
      active[1] = "XY"
      active[2] = "XY"
      active[3] = "XY"      
      
      previous_df = dframe.copy() 
      print(previous_df)
      for i in quadr:          
        previous_df.loc[previous_df['quadrant_local'] == i, 'fish_id'] = [x for x in ['X', 'Y']]
        #histograms_X_Y = {"X" + i:deque(maxlen=60), "Y" + i:deque(maxlen=60)}
        #list_idx = previous_df.loc[previous_df['quadrant_local'] == quadr].index.tolist()"""
      
          
      
    ########################################################################################################
    
    #block 2
    # check if there is only one fish (overlap event), and if yes, recreate the XY and change to it
    unique_quadrants = dframe.quadrant_local.unique()
    for index_q, row_q in enumerate(unique_quadrants):      
      fish_per_quadrant = (dframe['quadrant_local']==row_q).sum() 
      if fish_per_quadrant < 2:                   
        active[row_q] = "XY"
        is_first_iteraction[row_q] = True        
        histograms_X_Y["X" + str(row_q)] = deque(maxlen=60)
        histograms_X_Y["Y" + str(row_q)] = deque(maxlen=60)
            
        continue
      
        
        
        #########################################################################
        # block 1
        # # first interaction (frame) only after a collision event, so we have to reset XY    
             
      if (active[row_q] == "XY" and is_first_iteraction[row_q] == True) | (len(histograms_ids[str(1) + str(row_q)]) == 0):                       
        dframe.loc[dframe['quadrant_local'] == row_q, 'fish_id'] = [x for x in ['X', 'Y']]
        list_idx = dframe.loc[dframe['quadrant_local'] == row_q].index.tolist()                   
        histograms_X_Y['X'+ str(row_q)].append(histograms[list_idx[0]])  
        histograms_X_Y['Y'+ str(row_q)].append(histograms[list_idx[1]])             
        is_first_iteraction[row_q] = False      
              
      else:  
        filtered_df = dframe[(dframe.quadrant_local == row_q)]
        filtered_df.index = filtered_df.index.set_names(['original_index'])
        filtered_df = filtered_df.reset_index()      

        
        ########################################################################
        #### block 2
        # here, while the script is in 1-2 mode or xy mode, the script decide which fish is which based on last position
        #hpwever, if only 1 value in histograms_X_Y, means that we don´t have previous df, as it is the begining of xy, but xy was already added           
              
        
        previous_fish_1 = previous_df.loc[previous_df['quadrant_local'] == row_q].iloc[0]   
        previous_fish_2 = previous_df.loc[previous_df['quadrant_local'] == row_q].iloc[1]  

        previous_fish_1_id = previous_fish_1['fish_id']
        previous_fish_2_id = previous_fish_2['fish_id']
                
        previous_fish_1_position = previous_fish_1['position_fish_local']
        previous_fish_2_position = previous_fish_2['position_fish_local']

        previous_fish_1_area = previous_fish_1['fish_area']
        previous_fish_2_area = previous_fish_2['fish_area']        
      
        for idx, row in filtered_df.iterrows():
          
          cnt = contours[row['cnt_idx']]         
          
          distances_indices = [] 
          current_position_fish_local = row['position_fish_local'] 
          
          distance_value_1 = math.sqrt(   (previous_fish_1_position[0]-current_position_fish_local[0])**2 + (previous_fish_1_position[1]-current_position_fish_local[1])**2    )
          distances_indices.append(distance_value_1)
          distance_value_2 = math.sqrt(   (previous_fish_2_position[0]-current_position_fish_local[0])**2 + (previous_fish_2_position[1]-current_position_fish_local[1])**2    )
                  
          distances_indices.append(distance_value_2)
          distances_indices.append(distance_value_1)              
                  
          lower_indice = sorted(range(len(distances_indices)), key = lambda sub: distances_indices[sub])[:1]              
          
          if lower_indice[0] == 0:            
            dframe.loc[row['original_index'],'fish_id'] = previous_fish_1_id
          else:            
            dframe.loc[row['original_index'],'fish_id'] = previous_fish_2_id        
      
            #####################################################################################################################
            ####  block 3
            # here, the script add values to the history list with some filtering. Can be XY or Id.         
          
              
          if active[row_q] == "XY":        
            #histograms_X_Y[row['fish_id'] + str(row_q)].append(histograms[int(row['original_index'])])
            histograms_X_Y[str(dframe['fish_id'][row['original_index']]) + str(row_q)].append(histograms[int(row['original_index'])])          

            
            
    
      #only go ahead to choose which fish is which if the variable active = xy (means that it is time to choose)
      
      if active[row_q] == 'XY' and len(histograms_X_Y['Y' + str(row_q)]) == 60 and len(histograms_X_Y['X' + str(row_q)]) == 60:
      
        t_stat_filter, p_val_filter = stats.ttest_ind(histograms_X_Y['X' + str(row_q)], histograms_X_Y['Y' + str(row_q)], equal_var=False)
        cov = lambda x: np.std(x, ddof=1) / np.mean(x) * 100                   
        cov1 = cov(histograms_X_Y['Y' + str(row_q)])
        cov2 = cov(histograms_X_Y['X' + str(+row_q)])
        
            
        if p_val_filter < 0.01 and abs(t_stat_filter) > 10 and kurtosis(histograms_X_Y['Y' + str(row_q)]) < 1 and kurtosis(histograms_X_Y['Y' + str(row_q)]) > 0 and kurtosis(histograms_X_Y['X' + str(row_q)]) < 1 and kurtosis(histograms_X_Y['X' + str(row_q)]) > 0:  
          active[row_q] = 'id'
          
        else:
          #histograms_X_Y['Y'] = histograms_X_Y['Y']
          #histograms_X_Y['X'] = histograms_X_Y['X']
          continue
              
        
      ############################################################################################################
      # the minimum statistics and count (60) were reached in lists (XY), then we go ahead to choose which fish is which,
      #but if we have no id values yet, as is the very begining, we need to copy them from xy.      
        if len(histograms_ids['1' + str(row_q)]) < 60 or len(histograms_ids['2' + str(row_q)]) < 60:
          previous_fish_1 = previous_df.loc[previous_df['quadrant_local'] == row_q].iloc[0]   
          previous_fish_2 = previous_df.loc[previous_df['quadrant_local'] == row_q].iloc[1]
          previous_fish_1_id = previous_fish_1['fish_id']
          previous_fish_2_id = previous_fish_2['fish_id']
                   
          histograms_ids['1' + str(row_q)] = histograms_X_Y[previous_fish_1_id + str(row_q)]                    
          dframe.loc[(dframe.fish_id == previous_fish_1_id) & (dframe.quadrant_local == row_q), "fish_id"] = float(1) 
          histograms_ids['2' + str(row_q)] = histograms_X_Y[str(previous_fish_2_id) + str(row_q)]          
          dframe.loc[(dframe.fish_id == previous_fish_2_id) & (dframe.quadrant_local == row_q), "fish_id"] = float(2)              
        print("time to decide")    
      #############################################################################################################      
        # now we actually are going to decide which fish is which by statistcs          
        cov = lambda x: np.std(x, ddof=1) / np.mean(x) * 100
        t_stat_filter, p_val_filter = stats.ttest_ind(histograms_ids['1' + str(row_q)], histograms_ids['2' + str(row_q)], equal_var=False)          
        cov1 = cov(histograms_ids['1' + str(row_q)])
        cov2 = cov(histograms_ids['2' + str(row_q)])    
          
        current_fish = dframe.loc[dframe['quadrant_local'] == row_q]
        current_fish_0 = current_fish.iloc[0]
        current_fish_1 = current_fish.iloc[1]      

        templates_of_ID1 = histograms_ids['1' + str(row_q)]
        templates_of_ID2 = histograms_ids['2' + str(row_q)]    
            
        X_current_template_list = histograms_X_Y['X' + str(row_q)]
        Y_current_template_list = histograms_X_Y['Y' + str(row_q)]
        
        def reject_outliers(data, m=6.):
            d = np.abs(data - np.median(data))
            mdev = np.median(d)
            s = d / (mdev if mdev else 1.)
            return data[s < m].tolist()          
        
        #templates_of_ID1, templates_of_ID2, X_current_template_list, Y_current_template_list = reject_outliers(np.array(templates_of_ID1)), reject_outliers(np.array(templates_of_ID2)), reject_outliers(np.array(X_current_template_list)), reject_outliers(np.array(Y_current_template_list))     

        df1 = pd.DataFrame({'score': templates_of_ID1,
                  'group': 'id1'}) 
        
        #print(df1.size)
        #k2, p = stats.normaltest(templates_of_ID1)
        #print(p)
        #print(iqr(templates_of_ID1))
        df2 = pd.DataFrame({'score': templates_of_ID2,
                  'group': 'id2'}) 
        #print(df2.size)
        #k2, p = stats.normaltest(templates_of_ID2)
        #print(p)
        #print(iqr(templates_of_ID2))
        df3 = pd.DataFrame({'score': X_current_template_list,
                  'group': 'X'}) 
        #print(df3.size)
        #k2, p = stats.normaltest(X_current_template_list)
        #print(p)
      
        #sns.distplot(X_current_template_list)

        df4 = pd.DataFrame({'score': Y_current_template_list,
                  'group': 'Y'}) 
        #print(df4.size)
        #k2, p = stats.normaltest(Y_current_template_list)
        #print(p)
      
        #sns.distplot(Y_current_template_list)
        df = pd.concat([df1, df2, df3, df4])
        
        
        
        #stre = sns.boxplot(x="score", y="group", data=df)
        #sns.violinplot(x="score", y="group", data=df)
        #plt.show(block = False) 
                  
        #cv2.waitKey(0)
        
        tukey = sp.posthoc_ttest(df, val_col='score', group_col='group', p_adjust='holm')
        
        
        X1 = tukey['X'][0]
        X2 = tukey['X'][1]
        Y1 = tukey['Y'][0]
        Y2 = tukey['Y'][1]         
      
      
        minimum =  min(X1, X2, Y1, Y2)       
              
        if minimum == X2 or minimum == Y1:
          dframe.loc[(dframe.fish_id == 'X') & (dframe.quadrant_local == row_q), "fish_id"] = float(1) 
          dframe.loc[(dframe.fish_id == 'Y') & (dframe.quadrant_local == row_q), "fish_id"] = float(2) 
        else:
          dframe.loc[(dframe.fish_id == 'X') & (dframe.quadrant_local == row_q), "fish_id"] = float(2) 
          dframe.loc[(dframe.fish_id == 'Y') & (dframe.quadrant_local == row_q), "fish_id"] = float(1)
        
        pass            

      #####################################################################################   
    #block 6
    # Time to draw everything on a template
    drawn_image = blank_image.copy()
    drawn_image = cv2.drawContours(drawn_image, contours, -1, color=(255,0,0),thickness=-1)

    filt_dframe = dframe.loc[(dframe['fish_id'] == "X") | (dframe['fish_id'] == "Y") | (dframe['fish_id'] == 1.0) | (dframe['fish_id'] == 2.0)]
    filt_s_dataf = filt_dframe[['lenght_of_fish_local', 'position_fish_local', 'fish_tail_local', 'fish_head_local', 'quadrant_local', 'fish_area', 'fish_id']]
    filt_s_dataf.insert(loc=0,
          column='frame_number',
          value=idx_frame)
    
    filt_s_dataf.to_csv(path_to_save + "/" + file_name + '.csv', mode='a', index=False, header=False)
    # not append because it is in development now
  
        
    for c in idx_local[:8]:
      #tail plot
      cv2.circle(drawn_image, fish_tail_local[c], 2, (0, 0, 255), -1)

      #head
      cv2.circle(drawn_image, fish_head_local[c], 2, (0, 255, 0), -1)

      #center of mass
      cv2.circle(drawn_image, position_fish_local[c], 2, (0, 165, 255), -1)
      

    position_list = dframe.position_fish_local.tolist()
    fish_id = dframe.fish_id.tolist()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for indice, value in enumerate(fish_id):
     
      cv2.putText(drawn_image,str(fish_id[indice]),(position_list[indice]), font, 1,(255,255,255),2)
      

         
    cv2.line(drawn_image, (427, 0), (427,870), (255, 255, 255), thickness=1)
    cv2.line(drawn_image, (0, 417), (870,417), (255, 255, 255), thickness=1)   

   
    
    #cv2.rectangle(main, (int(x_y_line[0]), int(x_y_line[1])), (int(x_y_line[0])+100, int(x_y_line[1])+100), (36,255,12), 2)
      
    #show the image with filtered countours plotted
    imS = cv2.resize(drawn_image, (900, 900))              # Resize image   
    cv2.imshow('Frame',imS)

    previous_df = dframe.copy()  
    
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break



# When everything done, release the video capture object
cap.release()

# Closes all the frames
#cv2.destroy