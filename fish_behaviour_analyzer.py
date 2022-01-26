quadr = 'D'
import scipy.stats as stats
import cv2
import statistics
from cv2 import waitKey 
import numpy as np
import math
import pandas as pd
import imagehash
from PIL import Image     
import time
quad_D = 1
quad_D_count = 0
update_counter = 23
import matplotlib.pyplot as plt 
#import imutils
fish_0_history = []
fish_1_history = []
make_copy_last_seen = True
first = True

calculo_area = []
went_ok = 0
active = "id"
is_first_iteraction = False 
pd.set_option('display.max_columns', None)  
cap = cv2.VideoCapture('C:/Users/marci/Desktop/20191121_1454_iCab_L_C.avi')
background_image = cv2.imread('C:/Users/marci/Desktop/background_1.jpg')
bw_back = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
bw_back = cv2.GaussianBlur(bw_back, (9,9) ,0) 

#create a blank image to plot everything on
blank_image = np.zeros((bw_back.shape[0], bw_back.shape[1], 3), np.uint8)

previous_df = None
lopp = 0

previous_id_fish_local = []

histograms_ids = {1:[], 2:[]}

#cv2.imshow('background' , bw_back)

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
for idx_frame in range(11000,10000000,1):   #3000 to 4000
  print(idx_frame)
    
  
  cap.set(1, idx_frame)
  
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    
    first_fish = []
    second_fish = []

    imzz = cv2.resize(frame, (780, 780))              # Resize image   
    cv2.imshow('Main',imzz)
    #cv2.waitKey(1)
     
    
    
    original_img = frame.copy()

    # Display the resulting frame
    bw_mainImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bw_mainImage = cv2.GaussianBlur(bw_mainImage, (9,9) ,0)

    diff = cv2.absdiff(bw_back, bw_mainImage)
          
    ret,thresh = cv2.threshold(diff,15,255,cv2.THRESH_BINARY)  
    

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    

    lenght_of_fish_local = []
    position_fish_local = []
    quadrant_local = []
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
        
        
        #rect = cv2.minAreaRect(arr)
        #box = cv2.boxPoints(rect)
        #box = np.int0(box)
        #area_rec = cv2.contourArea(box)
        
        
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
        fish_COM = (aver_cm[0][0], aver_cm[0][1])       
                    

        fish_pectoral_lenght = math.sqrt( (aver_head[0] - aver_cm[0][0])  **2 + (aver_head[1] - aver_cm[0][1])**2    )

        final_template = frame[extTop[1]:extBot[1], extLeft[0]:extRight[0], :]        
        final_template = cv2.cvtColor(final_template, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([final_template], [0], None, [256], [0, 256])
        hist = hist[:185]
        hist_1 = hist[:146]       
        hist_2 = hist[146:]      
        final=sum(hist_1) + sum(hist_2)       
        final_grey=final[0]
       
        
       
        
      
        if (aver_cm[0][0] < 427 and aver_cm[0][1] < 417):  # belongs to quadrant A1
          quadrant_value = "A"
        elif (aver_cm[0][0] < 427 and aver_cm[0][1] > 417):
          quadrant_value = "B"
        elif (aver_cm[0][0] > 427 and aver_cm[0][1] < 417):
          quadrant_value = "C"
        else:
          quadrant_value = "D"       

          
      

        #store variables locally in the loop for immediate calculation purposes
        idx_local.append(counter)          
        lenght_of_fish_local.append(fish_pectoral_lenght)
        position_fish_local.append(fish_COM)
        fish_tail_local.append(list_of_points[max_index_tail])
        fish_head_local.append(aver_head)
        quadrant_local.append(quadrant_value)
        fish_area.append(area_rec)        
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
   
    
    #print(dframe)
    
    # sometimes fezes can be considered a fish, then filter by getting the biggest two contours per quadrant
    
    unique_quadrants = dframe.quadrant_local.unique()
    if previous_df is None:
      active = "XY"      
      previous_df = dframe.copy()           
      #previous_df.loc[previous_df['quadrant_local'] == quadr, 'fish_id'] = [x for x in [1, 2]]
      previous_df.loc[previous_df['quadrant_local'] == quadr, 'fish_id'] = [x for x in ['X', 'Y']]
      histograms_X_Y = {"X":[], "Y":[]}
      list_idx = previous_df.loc[previous_df['quadrant_local'] == quadr].index.tolist()

    
    for index_q, row_q in enumerate(unique_quadrants):      
      fish_per_quadrant = (dframe['quadrant_local']==quadr).sum()
      
    
      if row_q == quadr:

        #print(update_counter)    

        if fish_per_quadrant < 2:
          
       
                    
          active = "XY"
          is_first_iteraction = True         
          if make_copy_last_seen == True:
            dframe_last_seen = previous_df.copy()
            make_copy_last_seen = False         
          #histograms_last_seen = histograms_ids.copy() 
          histograms_X_Y = {"X":[], "Y":[]}
                
          continue

        
             
        if active == "XY" and is_first_iteraction == True: 
          
                         
          dframe.loc[dframe['quadrant_local'] == quadr, 'fish_id'] = [x for x in ['X', 'Y']]          
          list_idx = dframe.loc[dframe['quadrant_local'] == quadr].index.tolist()          
          histograms_X_Y['X'].append(histograms[list_idx[0]])
          histograms_X_Y['Y'].append(histograms[list_idx[1]])
          previous_df = dframe.copy()
          is_first_iteraction = False            
                        
         
               
       
        filtered_df = dframe[(dframe.quadrant_local == row_q)]
        filtered_df.index = filtered_df.index.set_names(['original_index'])
        filtered_df = filtered_df.reset_index()           
             
        for idx, row in filtered_df.iterrows():
          
          
          cnt = contours[row['cnt_idx']]
          
          
          
          distances_indices = [] 
          current_position_fish_local = row['position_fish_local']           
          
          previous_fish_1 = previous_df.loc[previous_df['quadrant_local'] == row_q].iloc[0]   
          previous_fish_2 = previous_df.loc[previous_df['quadrant_local'] == row_q].iloc[1]  
      
          previous_fish_1_id = previous_fish_1['fish_id']
          previous_fish_2_id = previous_fish_2['fish_id']
          
          previous_fish_1_position = previous_fish_1['position_fish_local']
          previous_fish_2_position = previous_fish_2['position_fish_local']

          previous_fish_1_area = previous_fish_1['fish_area']
          previous_fish_2_area = previous_fish_2['fish_area'] 
          
          distance_value_1 = math.sqrt(   (previous_fish_1_position[0]-current_position_fish_local[0])**2 + (previous_fish_1_position[1]-current_position_fish_local[1])**2    )
          distances_indices.append(distance_value_1)
          distance_value_2 = math.sqrt(   (previous_fish_2_position[0]-current_position_fish_local[0])**2 + (previous_fish_2_position[1]-current_position_fish_local[1])**2    )
          
          
          #let's decide which fish is which based on last position
          distances_indices.append(distance_value_2)
          distances_indices.append(distance_value_1)              
                  
          lower_indice = sorted(range(len(distances_indices)), key = lambda sub: distances_indices[sub])[:1]
      
                   
                       
        
          if lower_indice[0] == 0:
            dframe.loc[row['original_index'],'fish_id'] = previous_fish_1_id         
            
            #we need append to template history in case the fish area is higher than a threshold
            
            
                   
            if active == "XY":  
              histograms_X_Y[previous_fish_1_id].append(histograms[row['original_index']]) #((np.add((np.multiply(previous_histograms_X_Y[previous_fish_1_id], 3)), histograms[row['original_index']]) / 4)).astype(np.float32)
              if len(histograms_X_Y[previous_fish_1_id]) > 60:
                histograms_X_Y[previous_fish_1_id] = histograms_X_Y[previous_fish_1_id][-60:]
              
            else:
               
              histograms_ids[int(previous_fish_1_id)].append(histograms[row['original_index']]) #((np.add((np.multiply(previous_histograms_ids[int(previous_fish_1_id)], 3)), histograms[row['original_index']]) / 4)).astype(np.float32)
                                
              if len(histograms_ids[int(previous_fish_1_id)]) > 60:
                histograms_ids[int(previous_fish_1_id)] = histograms_ids[int(previous_fish_1_id)][-60:]
          
              
          else: #the same as above, but in a inverse way
            dframe.loc[row['original_index'],'fish_id'] = previous_fish_2_id          
            
              
            if active == "XY":    
              histograms_X_Y[previous_fish_2_id].append(histograms[row['original_index']]) 
              if len(histograms_X_Y[previous_fish_2_id]) > 60:
                histograms_X_Y[previous_fish_2_id] = histograms_X_Y[previous_fish_2_id][-60:]
              
            else:
              
              histograms_ids[int(previous_fish_2_id)].append(histograms[row['original_index']])                
              if len(histograms_ids[int(previous_fish_2_id)]) > 60:
                histograms_ids[int(previous_fish_2_id)] = histograms_ids[int(previous_fish_2_id)][-60:]
            
           
             
          
                  
        if active == "XY":
          previous_histograms_X_Y = histograms_X_Y.copy()
          if len(histograms_X_Y[previous_fish_1_id]) == 60 and len(histograms_X_Y[previous_fish_2_id]) == 60 and len(histograms_ids[1]) == 60 and len(histograms_ids[2]) == 60:          
            t_stat, p_val_first = stats.ttest_ind(histograms_X_Y[previous_fish_1_id], histograms_X_Y[previous_fish_2_id], equal_var=True)
            cov = lambda x: np.std(x, ddof=1) / np.mean(x) * 100
            cov1 = cov(histograms_X_Y[previous_fish_1_id])
            cov2 = cov(histograms_X_Y[previous_fish_2_id])            
            best_cov1 = cov(histograms_ids[1])
            best_cov2 = cov(histograms_ids[2])
            if p_val_first > 0.05 or cov1*0.80 > best_cov1 or cov2*0.80 > best_cov2:
              continue                       
       
          else:
            # test to see if xy std is higher than the template std, And if yes, don´t go ahead and keep appending values to xy
            if len(histograms_X_Y[previous_fish_1_id]) < 60 or len(histograms_X_Y[previous_fish_2_id]) < 60:
              continue
           
            
        else:          
          continue
        
        active = "id"
        
        

        
        
        # here we update our templates accordingly to standard deviation        
        if len(histograms_ids[1]) == 60 and len(histograms_ids[2]) == 60: 
                 
          cov = lambda x: np.std(x, ddof=1) / np.mean(x) * 100
          best_cov1 = cov(best_histogram_ids_1)
          best_cov2 = cov(best_histogram_ids_2)
          cov1 = cov(histograms_ids[1])
          cov2 = cov(histograms_ids[2])
          
          print(cov1)
          print(cov2)
 
          if cov1 > best_cov1 and cov2 > best_cov2:
            histograms_ids[1] = best_histogram_ids_1
            histograms_ids[2] = best_histogram_ids_2            
          else:
            best_histogram_ids_1 = histograms_ids[1]
            best_histogram_ids_2 = histograms_ids[2]         
            
        
        
        else:    
          #means that is the begining and we don't have enough ids yet, then we need to copy from best template XY
          histograms_ids[1] = histograms_X_Y[previous_fish_1_id]
          best_histogram_ids_1 = histograms_X_Y[previous_fish_1_id].copy()        
          dframe.loc[dframe.fish_id == previous_fish_1_id, "fish_id"] = float(1) 
          histograms_ids[2] = histograms_X_Y[previous_fish_2_id]
          best_histogram_ids_2 = histograms_X_Y[previous_fish_2_id].copy()
          dframe.loc[dframe.fish_id == previous_fish_2_id, "fish_id"] = float(2)        
          
          continue    
            
        
     

          
        current_fish = dframe.loc[dframe['quadrant_local'] == row_q]
        current_fish_0 = current_fish.iloc[0]
        current_fish_1 = current_fish.iloc[1]
        

       

        templates_of_ID1 = histograms_ids[1]
        templates_of_ID2 = histograms_ids[2]
        
      
                
        X_current_template_list = histograms_X_Y['X']
        if len(X_current_template_list) > 60:
          X_current_template_list = X_current_template_list[-60:]
                    
        Y_current_template_list = histograms_X_Y['Y']
        if len(Y_current_template_list) > 60:
          Y_current_template_list = Y_current_template_list[-60:]       

        
        t_stat, p_val = stats.ttest_ind(templates_of_ID1 + X_current_template_list, templates_of_ID2 + Y_current_template_list, equal_var=False)
        option1 = p_val
        t_stat, p_val = stats.ttest_ind(templates_of_ID1 + Y_current_template_list, templates_of_ID2 + X_current_template_list, equal_var=False)
        option2 = p_val
        
        if option1<option2:
          option = '1X'
        else:
          option = '1Y'

        

        

        ########### hash images ##########
       
           
        
               
        if option == '1X':
          dframe.loc[dframe.fish_id == 'X', "fish_id"] = float(1) 
          dframe.loc[dframe.fish_id == 'Y', "fish_id"] = float(2) 
        else:
          dframe.loc[dframe.fish_id == 'X', "fish_id"] = float(2) 
          dframe.loc[dframe.fish_id == 'Y', "fish_id"] = float(1) 

    
    drawn_image = blank_image.copy()
    drawn_image = cv2.drawContours(drawn_image, contours, -1, color=(255,0,0),thickness=-1)


      
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


    #show the image with filtered countours plotted
    imS = cv2.resize(drawn_image, (960, 540))              # Resize image   
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


