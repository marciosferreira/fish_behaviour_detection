quadr = 'D'
import scipy.stats as stats
import scikit_posthocs as sp
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import iqr
import cv2
from skimage.morphology import medial_axis
from skimage.morphology import skeletonize

import statistics
import seaborn as sns
from cv2 import waitKey 
import numpy as np
import math
import pandas as pd
import imagehash
from PIL import Image     
import time
quad_D = 1
quad_D_count = 0
#update_counter = 23
import matplotlib.pyplot as plt 
#import imutils
fish_0_history = []
fish_1_history = []
make_copy_last_seen = True
first = True
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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
for idx_frame in range(5700,10000000,1):   #3000 to 4000
  print(idx_frame)
    
  
  cap.set(1, idx_frame)
  
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    
    first_fish = []
    second_fish = []

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
          
    ret,thresh = cv2.threshold(diff,10,255,cv2.THRESH_BINARY)  
    

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
    skeleton_list = []
    
    for idx, cnt in enumerate(contours):
                  
      area = cv2.contourArea(cnt) 
      
      #calculate aspect ratio for template quality filtering
      
        

      if area > 200 and area < 1500:
        
        drawn_image_for_skeleton = blank_image.copy()
        drawn_image_for_skeleton = cv2.drawContours(drawn_image_for_skeleton, [cnt], -1, color=(255,255,255),thickness=-1)
        
        #cv2.imshow("skl", drawn_image_for_skeleton)
        bw_mainImage_sk = cv2.cvtColor(drawn_image_for_skeleton, cv2.COLOR_BGR2GRAY)
        # Creating kernel
        #kernel = np.ones((5, 5), np.uint8)  
        # Using cv2.erode() method 
        #image = cv2.erode(bw_mainImage_sk, kernel) 
        #image2 = cv2.dilate(image, kernel)
        
        #cv2.imshow("skl_final_eroded", bw_mainImage_sk)

        binarizedImage = bw_mainImage_sk / 255

        #skeleton = medial_axis(bw_mainImage_sk).astype(np.uint8)
        skeleton = skeletonize(binarizedImage)
        
        #fig, axes = plt.subplots(1, 1, figsize=(18, 10), sharex=True, sharey=True)
        #ax = axes.ravel()

        #plt.imshow(skeleton, cmap=plt.cm.gray)
     
       

      
        #plt.show()
        skeleton = (skeleton*255).astype(np.uint8)
    
        #cv2.waitKey(0)              
        #skeleton_coords = np.argwhere(skeleton)
        skeleton_coords = list(zip(*np.nonzero(skeleton)))
        #nzCount = cv2.countNonZero(skeleton)
        
        #cv2.waitKey(0)
       
        #cv2.imshow("skl_final", skeleton)
       
                
              
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
  
        #cv2.waitKey(0)
        
        #sort skeleton by next coordinate, starting from head
        #############################
        #skeleton_coords = skeleton_coords.tolist() 
        '''print("xxxxxxxxxxx")
        print(skeleton_coords)
        distances_skeleton_to_head = []
        for l in skeleton_coords:                  
          distance = math.sqrt(   (aver_head[0]-l[1])**2 + (aver_head[1]-l[0])**2    )
          distances_skeleton_to_head.append(distance)
        
        #get farthest value
        min_value = min(distances_skeleton_to_head)
        min_index_skeleton = distances_skeleton_to_head.index(min_value)       
        min_skl_coord = skeleton_coords.pop(min_index_skeleton)           
        distances_skeleton_to_head.pop(min_index_skeleton)'''
        
       
        sorted_skeleton = []               
        while len(skeleton_coords) > 1:
          distances_skeleton_from_head = []
          for l in skeleton_coords:
           
            distance = math.sqrt(   (aver_head[0]-l[1])**2 + (aver_head[1]-l[0])**2    )
            distances_skeleton_from_head.append(distance)
          min_value = min(distances_skeleton_from_head)
          min_index_skeleton = distances_skeleton_from_head.index(min_value)
          min_skl_coord = skeleton_coords.pop(min_index_skeleton)          
          sorted_skeleton.append(min_skl_coord)
      
        #drawn_image_2 = frame.copy()
        #cv2.circle(drawn_image_2, (sorted_skeleton[0][1], sorted_skeleton[0][0]), 2, (0, 0, 255), -1)
        #cv2.imshow("testexxx", drawn_image_2 )
        #############################
        #print(sorted_skeleton)
        #drawn_image_2 = frame.copy()

        #cv2.drawContours(drawn_image_2, [skeleton], -1, color=(0,0,0),thickness=2)
        #cv2.circle(frame, (sorted_skeleton[-1][1], sorted_skeleton[-1][0]), 2, (0, 255, 0), -1)
        
        #cv2.imshow("teste", skeleton )
        #cv2.imshow("framesssss", frame )
        #print(skeleton_coords)
        #cv2.waitKey(0)

        
        

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
        histograms.append(fish_pectoral_lenght)  
        countours_idx.append(0)
        template_area.append(area)
        template_blur.append(0)
        template_dark.append(0)
        skeleton_list.append(sorted_skeleton)       
                      
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
    
    unique_quadrants = dframe.quadrant_local.unique()
    if previous_df is None:     
      active = "XY"      
      previous_df = dframe.copy()           
      previous_df.loc[previous_df['quadrant_local'] == quadr, 'fish_id'] = [x for x in ['X', 'Y']]
      histograms_X_Y = {"X":[], "Y":[]}
      list_idx = previous_df.loc[previous_df['quadrant_local'] == quadr].index.tolist()
      
    ########################################################################################################
    
    #block 2
    # check if there is only one fish (overlap event), and if yes, recreate the XY and change to it
    
    for index_q, row_q in enumerate(unique_quadrants):      
      fish_per_quadrant = (dframe['quadrant_local']==quadr).sum()
      
    
      if row_q == quadr:   

        if fish_per_quadrant < 2:      
                    
          active = "XY"
          is_first_iteraction = True         
          if make_copy_last_seen == True:
            dframe_last_seen = previous_df.copy()
            make_copy_last_seen = False         
          histograms_X_Y = {"X":[], "Y":[]}
                
          continue
        
        
        #########################################################################
        # block 1
        # the initial xy finish and the fish don´t have id as history, then just attribute the xy history for the id history
   
             
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
  
        
        ########################################################################
        #### block 2
        # here, while the script is in 1-2 mode or xy mode, the script decide which fish is which based on last position           
             
        
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
          
         
        for idx, row in filtered_df.iterrows():         
          print(active)
          if active == "XY":
            print(dframe['fish_id'][row['original_index']])                          
            histograms_X_Y[dframe['fish_id'][row['original_index']]].append(histograms[int(row['original_index'])])               
          
        #only go ahead to choose which fish is which if the variable active = xy (means that it is time to choose)
        print(len(histograms_X_Y['X']))
        print(len(histograms_X_Y['Y']))
        if active == 'XY' and len(histograms_X_Y['Y']) == 60 and len(histograms_X_Y['X']) == 60:
          t_stat_filter, p_val_filter = stats.ttest_ind(histograms_X_Y['X'], histograms_X_Y['Y'], equal_var=False)
          cov = lambda x: np.std(x, ddof=1) / np.mean(x) * 100                   
          cov1 = cov(histograms_X_Y['Y'])
          cov2 = cov(histograms_X_Y['X'])
          print(kurtosis(histograms_X_Y['X']))
          print(kurtosis(histograms_X_Y['Y']))
          print(cov2)
          print(p_val_filter)
          print(t_stat_filter)
             
          if p_val_filter < 0.01 and abs(t_stat_filter) > 10 : #and kurtosis(histograms_X_Y['Y']) < 1 and kurtosis(histograms_X_Y['Y']) > 0 and kurtosis(histograms_X_Y['X']) < 1 and kurtosis(histograms_X_Y['X']) > 0:  
            active = 'id'
          else:
            histograms_X_Y['Y'] = histograms_X_Y['Y'][-59:]
            histograms_X_Y['X'] = histograms_X_Y['X'][-59:]
            break
                
          
        ############################################################################################################
        # the minimum values were reached in lists (id and XY), then we go ahead to choose which fish is which,
        #but if we have no id values yet, as is the very begining, we need to copy them from xy.      
          if len(histograms_ids[1]) < 60 or len(histograms_ids[2]) < 60:           
            histograms_ids[1] = histograms_X_Y[previous_fish_1_id]                    
            dframe.loc[dframe.fish_id == previous_fish_1_id, "fish_id"] = float(1) 
            histograms_ids[2] = histograms_X_Y[previous_fish_2_id]          
            dframe.loc[dframe.fish_id == previous_fish_2_id, "fish_id"] = float(2)              
              
        #############################################################################################################      
          # now we actually are going to decide which fish is which by statistcs          
          cov = lambda x: np.std(x, ddof=1) / np.mean(x) * 100
          t_stat_filter, p_val_filter = stats.ttest_ind(histograms_ids[1], histograms_ids[2], equal_var=False)          
          cov1 = cov(histograms_ids[1])
          cov2 = cov(histograms_ids[2])        
            
          current_fish = dframe.loc[dframe['quadrant_local'] == row_q]
          current_fish_0 = current_fish.iloc[0]          
          current_fish_1 = current_fish.iloc[1]
               

          templates_of_ID1 = histograms_ids[1]
          templates_of_ID1 = np.array(templates_of_ID1)
    
          
          templates_of_ID2 = histograms_ids[2]
          templates_of_ID2 = np.array(templates_of_ID2)
         
      
          X_current_template_list = histograms_X_Y['X']
          X_current_template_list = np.array(X_current_template_list)  
          
          Y_current_template_list = histograms_X_Y['Y']
          Y_current_template_list = np.array(Y_current_template_list)
          
          
          
          def reject_outliers(data, m=6.):
              d = np.abs(data - np.median(data))
              mdev = np.median(d)
              s = d / (mdev if mdev else 1.)
              return data[s < m].tolist()          
          
          #templates_of_ID1, templates_of_ID2, X_current_template_list, Y_current_template_list = reject_outliers(np.array(templates_of_ID1)), reject_outliers(np.array(templates_of_ID2)), reject_outliers(np.array(X_current_template_list)), reject_outliers(np.array(Y_current_template_list))     

          df1 = pd.DataFrame({'score': templates_of_ID1,
                    'group': 'id1'}) 
          print("data")
          #print(df1.size)
          k2, p = stats.normaltest(templates_of_ID1)
          #print(p)
          #print(iqr(templates_of_ID1))
          df2 = pd.DataFrame({'score': templates_of_ID2,
                    'group': 'id2'}) 
          #print(df2.size)
          k2, p = stats.normaltest(templates_of_ID2)
          #print(p)
          #print(iqr(templates_of_ID2))
          df3 = pd.DataFrame({'score': X_current_template_list,
                    'group': 'X'}) 
          #print(df3.size)
          k2, p = stats.normaltest(X_current_template_list)
          #print(p)
          
          print("kurtosis")
          print(kurtosis(X_current_template_list))
          #sns.distplot(X_current_template_list)

          df4 = pd.DataFrame({'score': Y_current_template_list,
                    'group': 'Y'}) 
          #print(df4.size)
          k2, p = stats.normaltest(Y_current_template_list)
          #print(p)
          print("kurtosis")
          print(kurtosis(Y_current_template_list))
          #sns.distplot(Y_current_template_list)
          df = pd.concat([df1, df2, df3, df4])
          
          
          
          #stre = sns.boxplot(x="score", y="group", data=df)
          sns.violinplot(x="score", y="group", data=df)
          plt.show() 
                   
          #cv2.waitKey(0)
          
          tukey = sp.posthoc_ttest(df, val_col='score', group_col='group', p_adjust='holm')
          
          
          X1 = tukey['X'][0]
          X2 = tukey['X'][1]
          Y1 = tukey['Y'][0]
          Y2 = tukey['Y'][1]         
        
        
          minimum =  min(X1, X2, Y1, Y2)       
                
          if minimum == X2 or minimum == Y1:
            dframe.loc[dframe.fish_id == 'X', "fish_id"] = float(1) 
            dframe.loc[dframe.fish_id == 'Y', "fish_id"] = float(2) 
          else:
            dframe.loc[dframe.fish_id == 'X', "fish_id"] = float(2) 
            dframe.loc[dframe.fish_id == 'Y', "fish_id"] = float(1)          

      #####################################################################################   
    #block 6
    # Time to draw everything on a template
    drawn_image = frame.copy()
    #drawn_image = cv2.drawContours(drawn_image, contours, -1, color=(255,0,0),thickness=-1)


      
    for c in idx_local[:8]:
      
      #tail plot
      #cv2.circle(drawn_image, fish_tail_local[c], 2, (0, 0, 255), -1)
      lenght = len(skeleton_list[c])      
      step = int(lenght*.70/4)
      
        
      
      for x in range(-1, -step*4, -(step)):
        #print(skeleton_list[c][x])
        
        cv2.circle(drawn_image, (skeleton_list[c][x][1], skeleton_list[c][x][0]), 2, (0, 0, 255), -1)

      #head
      cv2.circle(drawn_image, fish_head_local[c], 2, (0, 255, 0), -1)

      #center of mass
      cv2.circle(drawn_image, position_fish_local[c], 2, (0, 165, 255), -1)
      
      
      

    position_list = dframe.position_fish_local.tolist()
    fish_id = dframe.fish_id.tolist()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for indice, value in enumerate(fish_id):
     
      cv2.putText(drawn_image,str(fish_id[indice]),(position_list[indice]), font, 0.5,(0,0,0),1)
    

         
    cv2.line(drawn_image, (427, 0), (427,870), (255, 255, 255), thickness=1)
    cv2.line(drawn_image, (0, 417), (870,417), (255, 255, 255), thickness=1)   


    #show the image with filtered countours plotted
    imS = cv2.resize(drawn_image, (880, 880))              # Resize image   
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


