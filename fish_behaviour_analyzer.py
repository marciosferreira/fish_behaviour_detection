quadr = 'B'
import cv2
import numpy as np
import math
import pandas as pd
import time
quad_D = 1
quad_D_count = 0
update_counter = 3
import matplotlib.pyplot as plt 
import imutils
fish_0_history = []
fish_1_history = []
make_copy_last_seen = True
first = True
went_ok = 0
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

histograms_ids = ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan']

#cv2.imshow('background' , bw_back)

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
for idx_frame in range(3900,10000000,1):   #3000 to 4000
  print(idx_frame)
    
  
  cap.set(1, idx_frame)
  
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    
    first_fish = []
    second_fish = []

    imzz = cv2.resize(frame, (780, 780))              # Resize image   
    cv2.imshow('Main',imzz)
    cv2.waitKey(1)
     
    
    
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

    counter = 0
    
    for idx, cnt in enumerate(contours):
                  
      area = cv2.contourArea(cnt)        


      if area > 100 and area < 1500:
        
        
        image = frame 
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask = cv2.drawContours(mask, [cnt], -1, color=(255,255,255),thickness=-1)          
        result1 = cv2.bitwise_and(image, mask)
        hsv_img = cv2.cvtColor(result1, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]
        hist_final = cv2.calcHist([h],[0],None,[256],[0,256])
        hist_final = hist_final[1:70]
        
        #plt.subplot(2, 2, 1) # row 1, col 2 index 1
        #plt.plot(hist_final, color='r', label="h")          
        #plt.title('v')

        #plt.subplot(2, 2, 2) # index 2
        #plt.plot(hist_final, color='r', label="h")
        #plt.show()
        
        
        
        
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
        aver_head = (aver_head[0][0], aver_head[0][1])


        #calculate the center of mass of the fish excluding the tail
        distances_cm = []
        for j in cnt:
                  
          #append distances betwen head and each point of the fish
          distance = math.sqrt(   (j[0][0]-int(aver_head[0]))**2 + (j[0][1]-int(aver_head[1]))**2    )
          distances_cm.append(distance)            

        slowest_indices = sorted(range(len(distances_cm)), key = lambda sub: distances_cm[sub])[:int(fish_total_pixels*0.8)]
        slowest_indices_for_template = sorted(range(len(distances_cm)), key = lambda sub: distances_cm[sub])[:int(fish_total_pixels)]                   
        
            
        nearests_values = []
        nearest_values_for_template = []
        
        for e in slowest_indices:
          list_value_cm = cnt[e].tolist()              
          nearests_values.append(list_value_cm)

        arr = np.array(nearests_values)
        
        for e in slowest_indices_for_template:
          if distances_cm[e] < 35:
            list_value_cm = cnt[e].tolist()              
            nearest_values_for_template.append(list_value_cm)

        arr_for_template = np.array(nearest_values_for_template)
        
        contours_body = arr_for_template
        #is the nearest_values the fish without the tail?
        
        #canvas_for_body = np.zeros(frame.shape, dtype=original_img.dtype)            
        #mask_template_body = cv2.drawContours(canvas_for_body, [arr], -1, color=(255,255,255),thickness=-1)
        #mask_template_body = cv2.GaussianBlur(mask_template_body, (1,1) ,0)

        #bw_body_img = cv2.cvtColor(mask_template_body, cv2.COLOR_BGR2GRAY)               
        #ret,thresh = cv2.threshold(bw_body_img,15,255,cv2.THRESH_BINARY)            
                
        #contours_body, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #contours_body = contours_body[0]
        leftmost = tuple(contours_body[contours_body[:,:,0].argmin()][0])
        rightmost = tuple(contours_body[contours_body[:,:,0].argmax()][0])
        topmost = tuple(contours_body[contours_body[:,:,1].argmin()][0])
        bottommost = tuple(contours_body[contours_body[:,:,1].argmax()][0])
        rightmost = tuple(contours_body[contours_body[:,:,0].argmax()][0])
        #canvas = np.zeros(frame.shape, dtype=original_img.dtype)        
        #canvas = np.zeros((bottommost-topmost+40, rightmost-leftmost+40, 3), dtype=original_img.dtype)
        #mask_template = cv2.drawContours(canvas, [contours_body], -1, color=(255,255,255),thickness=-1)          
        #mask_template_cut = cv2.bitwise_and(frame, mask_template)
        #black_pixels = np.where((mask_template_cut[:, :, 0] == 0) & (mask_template_cut[:, :, 1] == 0) & (mask_template_cut[:, :, 2] == 0))
        # set those pixels to white based on Boolean mask
        #mask_template_cut[black_pixels] = [255, 255, 255] #[200, 202, 199]    
        #original_img = frame.copy()
        #mask3 = cv2.drawContours(template, [cnt], -1, color=(255,255,255),thickness=-1)   
        template = frame[topmost[1]:bottommost[1], leftmost[0]:rightmost[0]]
        
       
        #imzzx = cv2.resize(template, (780, 780)) 
        #cv2.imshow('template',template)
        #mask_template_body_final = cv2.drawContours(canvas_for_body, contours_b, -1, color=(0,255,0),thickness=-1)
        #print(len(contours_b))
        #imzzxx = cv2.resize(mask_template_body_final, (780, 780))
        #cv2.imshow("testatando", imzzxx)
        #cv2.waitKey(0) 
        
        
        #mask_template_body_final = cv2.drawContours(canvas_for_body, [contours[0][0]], -1, color=(0,255,0),thickness=-1)

        #imzzxx = cv2.resize(mask_template_body_final, (780, 780))
        #cv2.imshow("testatando", imzzxx)
        #cv2.waitKey(0)     
        
        aver_cm = np.mean(arr, axis=0).astype(int)        

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
        position_fish_local.append((aver_cm[0][0], aver_cm[0][1]))
        fish_tail_local.append(list_of_points[max_index_tail])
        fish_head_local.append(aver_head)
        quadrant_local.append(quadrant_value)
        fish_area.append(area)
        histograms.append(template)   # need to fix afterwards
        countours_idx.append(idx)
        
        
        
                      
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
    #dframe['fish_color'] = np.nan
    

    if previous_df is None:
      
      previous_df = dframe.copy()
      previous_histograms = histograms.copy()
      previous_histograms_ids = histograms_ids.copy()     
     
      previous_df.loc[previous_df['quadrant_local'] == quadr, 'fish_id'] = [x for x in [1, 2]]
      list_idx = previous_df.loc[previous_df['quadrant_local'] == quadr].index.tolist()
         
      previous_histograms_ids[1] = histograms[list_idx[0]]
      previous_histograms_ids[2] = histograms[list_idx[1]]

       
    unique_quadrants = dframe.quadrant_local.unique()
    
    for index_q, row_q in enumerate(unique_quadrants):      
      fish_per_quadrant = (dframe['quadrant_local']==quadr).sum()

      if row_q == quadr:

        #print(update_counter)    

        if fish_per_quadrant < 2:          
          update_counter = 0          
          if make_copy_last_seen == True:
            dframe_last_seen = previous_df.copy()
            make_copy_last_seen = False         
          histograms_last_seen = histograms_ids.copy() 
          histograms_X_Y = {"X": 'nan', "Y": 'nan'}
                
          continue

        
             
        if update_counter == 1:                    
          dframe.loc[dframe['quadrant_local'] == quadr, 'fish_id'] = [x for x in ['X', 'Y']]          
          list_idx = dframe.loc[dframe['quadrant_local'] == quadr].index.tolist()          
          histograms_X_Y['X'] = histograms[list_idx[0]]
          histograms_X_Y['Y'] = histograms[list_idx[1]]
          previous_df = dframe.copy()         
          previous_histograms_X_Y = histograms_X_Y.copy()                 
         
               
       
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
            
            #take the oportunity to update the fish area (area is not used for a while)
            dframe.loc[row['original_index'],'fish_area'] = (previous_fish_1_area * 40 + row['fish_area'])/21
            
            #update the histograms
            if update_counter  <= 2:  
              histograms_X_Y[previous_fish_1_id] = histograms[row['original_index']] #((np.add((np.multiply(previous_histograms_X_Y[previous_fish_1_id], 3)), histograms[row['original_index']]) / 4)).astype(np.float32)
            else:
              histograms_ids[int(previous_fish_1_id)] = histograms[row['original_index']] #((np.add((np.multiply(previous_histograms_ids[int(previous_fish_1_id)], 3)), histograms[row['original_index']]) / 4)).astype(np.float32)

              
          else: #the same as above, but in a inverse way
            dframe.loc[row['original_index'],'fish_id'] = previous_fish_2_id
            dframe.loc[row['original_index'],'fish_area'] = (previous_fish_2_area * 40 + row['fish_area'])/21
            
            #update the histograms
            if update_counter  <= 2:  
              histograms_X_Y[previous_fish_2_id] = histograms[row['original_index']] #((np.add((np.multiply(previous_histograms_X_Y[previous_fish_2_id], 3)), histograms[row['original_index']]) / 4)).astype(np.float32)
            else:
              histograms_ids[int(previous_fish_2_id)] = histograms[row['original_index']] #((np.add((np.multiply(previous_histograms_ids[int(previous_fish_2_id)], 3)), histograms[row['original_index']]) / 4)).astype(np.float32)

        if update_counter > 0 and update_counter < 2:
          previous_histograms_X_Y = histograms_X_Y.copy()  
        
        # here we decide which fish is 1 and 2 based on X and Y
        if update_counter == 2:
          #print(dframe)                   
          current_fish = dframe.loc[dframe['quadrant_local'] == row_q]
          current_fish_a = current_fish.iloc[0]
          current_fish_b = current_fish.iloc[1]
          
          current_fish_a_id = current_fish_a['fish_id']
          hist_of_a = histograms_X_Y[current_fish_a_id]
           
          current_fish_b_id = current_fish_b['fish_id']
          hist_of_b = histograms_X_Y[current_fish_b_id]
                  
          #hist_X = histograms_X_Y['X']    
          #hist_Y = histograms_X_Y['Y']
                      
          a_histo_last_seen = histograms_ids[1]
          cv2.imshow('a_histo_last_seen',a_histo_last_seen)
          #cv2.waitKey(0)  #need to be fixed to be dinamic
          '''temp_size_a = a_histo_last_seen.shape
          colum_add = np.full((temp_size_a[0], 100, 3), 255, dtype=original_img.dtype)
          #row_add = np.full((50, 498, 3), 255, dtype=original_img.dtype)
          #cropped_img = original_img[450:,450:]
          result_1 = np.append(a_histo_last_seen, colum_add, axis=1)
          result_2 = np.append(colum_add, result_1, axis=1)          
          temp_size = result_2.shape
          row_add = np.full((100, temp_size[1], 3), 255, dtype=original_img.dtype)         
          result_3 = np.append(result_2, row_add, axis=0)
          a_histo_last_seen = np.append(row_add, result_3, axis=0)  '''
          
          
                      
          b_histo_last_seen = histograms_ids[2]    #need to be fixed to be dinamic
          cv2.imshow('b_histo_last_seen', b_histo_last_seen)
          #cv2.waitKey(0)  #need to be fixed to be dinamic
          '''temp_size_a = b_histo_last_seen.shape
          colum_add = np.full((temp_size_a[0], 5, 3), 255, dtype=original_img.dtype)
          #row_add = np.full((50, 498, 3), 255, dtype=original_img.dtype)
          #cropped_img = original_img[450:,450:]
          result_1 = np.append(b_histo_last_seen, colum_add, axis=1)
          result_2 = np.append(colum_add, result_1, axis=1)          
          temp_size = result_2.shape
          row_add = np.full((5, temp_size[1], 3), 255, dtype=original_img.dtype)         
          result_3 = np.append(result_2, row_add, axis=0)
          b_histo_last_seen = np.append(row_add, result_3, axis=0)'''
       
          
                    
          temp_size_a = hist_of_a.shape
          colum_add = np.full((temp_size_a[0], 50, 3), 255, dtype=original_img.dtype)
          #row_add = np.full((50, 498, 3), 255, dtype=original_img.dtype)
          #cropped_img = original_img[450:,450:]
          result_1 = np.append(hist_of_a, colum_add, axis=1)
          result_2 = np.append(colum_add, result_1, axis=1)          
          temp_size = result_2.shape
          row_add = np.full((50, temp_size[1], 3), 255, dtype=original_img.dtype)         
          result_3 = np.append(result_2, row_add, axis=0)
          image_ind_a = np.append(row_add, result_3, axis=0)
          #cv2.imshow('fff', image_ind_a)
          #cv2.waitKey(0)  
          #imzzx = cv2.resize(template, (780, 780)) 
          #cv2.imshow('Image',image_ind)
          #cv2.waitKey(0)  
          cv2.imshow('image_ind_a', image_ind_a)

          
          
          temp_size_b = hist_of_b.shape
          colum_add = np.full((temp_size_b[0], 50, 3), 255, dtype=original_img.dtype)
          #row_add = np.full((50, 498, 3), 255, dtype=original_img.dtype)
          #cropped_img = original_img[450:,450:]
          result_1 = np.append(hist_of_b, colum_add, axis=1)
          result_2 = np.append(colum_add, result_1, axis=1)          
          temp_size = result_2.shape
          row_add = np.full((50, temp_size[1], 3), 255, dtype=original_img.dtype)         
          result_3 = np.append(result_2, row_add, axis=0)
          image_ind_b = np.append(row_add, result_3, axis=0)  
          cv2.imshow('image_ind_b', image_ind_b)
          cv2.waitKey(0)
          
          similarity_acurrent_aaverage = []
          for i in range(1,361):            
            rotated_template = imutils.rotate_bound(a_histo_last_seen, i)
            #create a boolean mask from black pixels
            black_pixels = np.where((rotated_template[:, :, 0] == 0) & (rotated_template[:, :, 1] == 0) & (rotated_template[:, :, 2] == 0))
            # set those pixels to white based on Boolean mask
            rotated_template[black_pixels] = [255, 255, 255]           
            #cv2.imshow("cut", original_img[450:,450:] )
            #cv2.waitKey(1)
            #z = np.zeros((420,50,3), dtype=original_img.dtype)
               # Resize image   
            #cv2.imshow('rotated',rotated_template) 
            #print(result.shape)
            #cv2.waitKey(1)       
                     
            result = cv2.matchTemplate(image_ind_a, rotated_template, cv2.TM_CCOEFF_NORMED)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
            (startX, startY) = maxLoc
            endX = startX + rotated_template.shape[1]
            endY = startY + rotated_template.shape[0]              
                       
            #final_img = cv2.rectangle(canvas, (startX, startY), (endX, endY), (255, 0, 0), 3)
            #imzz = cv2.resize(final_img, (780, 780))              # Resize image   
            #cv2.imshow('Final image',imzz)
            #cv2.waitKey(1)
            #print(idx)
            #print(maxVal)
            #if idx == 0:
            similarity_acurrent_aaverage.append(maxVal)
          import statistics 
       
          #similarity_acurrent_aaverage.sort()
          #similarity_acurrent_aaverage = similarity_acurrent_aaverage[-3:]
          #similarity_acurrent_aaverage = statistics.mean(similarity_acurrent_aaverage)
          similarity_acurrent_aaverage = max(similarity_acurrent_aaverage)
            
            
          similarity_bcurrent_baverage = []
          for i in range(1,361):            
            rotated_template = imutils.rotate_bound(b_histo_last_seen, i)
            #create a boolean mask from black pixels
            black_pixels = np.where((rotated_template[:, :, 0] == 0) & (rotated_template[:, :, 1] == 0) & (rotated_template[:, :, 2] == 0))
            # set those pixels to white based on Boolean mask
            rotated_template[black_pixels] = [255, 255, 255]           
            #cv2.imshow("cut", original_img[450:,450:] )
            #cv2.waitKey(1)
            #z = np.zeros((420,50,3), dtype=original_img.dtype)
               # Resize image   
            #cv2.imshow('rotated',rotated_template) 
            #print(result.shape)
            #cv2.waitKey(1)       
                     
            result = cv2.matchTemplate(image_ind_b, rotated_template, cv2.TM_CCOEFF_NORMED)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
            (startX, startY) = maxLoc
            endX = startX + rotated_template.shape[1]
            endY = startY + rotated_template.shape[0]              
                       
            #final_img = cv2.rectangle(canvas, (startX, startY), (endX, endY), (255, 0, 0), 3)
            #imzz = cv2.resize(final_img, (780, 780))              # Resize image   
            #cv2.imshow('Final image',imzz)
            #cv2.waitKey(1)
            #print(idx)
            #print(maxVal)
            #if idx == 0:
            similarity_bcurrent_baverage.append(maxVal)
            
          #similarity_bcurrent_baverage.sort() 
          #similarity_bcurrent_baverage = similarity_bcurrent_baverage[-3:]
          #similarity_bcurrent_baverage = statistics.mean(similarity_bcurrent_baverage)
          similarity_bcurrent_baverage = max(similarity_bcurrent_baverage)
     
          
          
          similarity_acurrent_baverage = []
          for i in range(1,361):            
            rotated_template = imutils.rotate_bound(b_histo_last_seen, i)
            #create a boolean mask from black pixels
            black_pixels = np.where((rotated_template[:, :, 0] == 0) & (rotated_template[:, :, 1] == 0) & (rotated_template[:, :, 2] == 0))
            # set those pixels to white based on Boolean mask
            rotated_template[black_pixels] = [255, 255, 255]           
            #cv2.imshow("cut", original_img[450:,450:] )
            #cv2.waitKey(1)
            #z = np.zeros((420,50,3), dtype=original_img.dtype)
               # Resize image   
            #cv2.imshow('rotated',rotated_template) 
            #print(result.shape)
            #cv2.waitKey(1)       
                     
            result = cv2.matchTemplate(image_ind_a, rotated_template, cv2.TM_CCOEFF_NORMED)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
            (startX, startY) = maxLoc
            endX = startX + rotated_template.shape[1]
            endY = startY + rotated_template.shape[0]              
                       
            #final_img = cv2.rectangle(canvas, (startX, startY), (endX, endY), (255, 0, 0), 3)
            #imzz = cv2.resize(final_img, (780, 780))              # Resize image   
            #cv2.imshow('Final image',imzz)
            #cv2.waitKey(1)
            #print(idx)
            #print(maxVal)
            #if idx == 0:
            similarity_acurrent_baverage.append(maxVal)
            
          #similarity_acurrent_baverage.sort()
          #similarity_acurrent_baverage = similarity_acurrent_baverage[-3:]
          #similarity_acurrent_baverage = statistics.mean(similarity_acurrent_baverage)
          similarity_acurrent_baverage = max(similarity_acurrent_baverage)
          
          
          
           
                
          similarity_bcurrent_aaverage = []
          for i in range(1,361):            
            rotated_template = imutils.rotate_bound(a_histo_last_seen, i)
            #create a boolean mask from black pixels
            black_pixels = np.where((rotated_template[:, :, 0] == 0) & (rotated_template[:, :, 1] == 0) & (rotated_template[:, :, 2] == 0))
            # set those pixels to white based on Boolean mask
            rotated_template[black_pixels] = [255, 255, 255]           
            #cv2.imshow("cut", original_img[450:,450:] )
            #cv2.waitKey(1)
            #z = np.zeros((420,50,3), dtype=original_img.dtype)
               # Resize image   
            #cv2.imshow('rotated',rotated_template) 
            #print(result.shape)
            #cv2.waitKey(1)       
                     
            result = cv2.matchTemplate(image_ind_b, rotated_template, cv2.TM_CCOEFF_NORMED)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
            (startX, startY) = maxLoc
            endX = startX + rotated_template.shape[1]
            endY = startY + rotated_template.shape[0]              
                       
            #final_img = cv2.rectangle(canvas, (startX, startY), (endX, endY), (255, 0, 0), 3)
            #imzz = cv2.resize(final_img, (780, 780))              # Resize image   
            #cv2.imshow('Final image',imzz)
            #cv2.waitKey(1)
            #print(idx)
            #print(maxVal)
            #if idx == 0:
            similarity_bcurrent_aaverage.append(maxVal)
            
          #similarity_bcurrent_aaverage.sort()
          #similarity_bcurrent_aaverage = similarity_bcurrent_aaverage[-3:]
          #similarity_bcurrent_aaverage = statistics.mean(similarity_bcurrent_aaverage)
          similarity_bcurrent_aaverage = max(similarity_bcurrent_aaverage)
                           
          
          print(similarity_acurrent_aaverage)
          print(similarity_acurrent_baverage)
          print(similarity_bcurrent_aaverage)
          print(similarity_bcurrent_baverage)
          
          simil_list = [similarity_acurrent_aaverage, similarity_acurrent_baverage, similarity_bcurrent_aaverage, similarity_bcurrent_baverage]
          max_simil = max(simil_list)
          print(max_simil)
          
          if max_simil == similarity_acurrent_aaverage or max_simil == similarity_bcurrent_baverage:
            dframe.loc[dframe.fish_id == current_fish_a_id, "fish_id"] = float(1) # needs to be fixed to be dynamic
            dframe.loc[dframe.fish_id == current_fish_b_id, "fish_id"] = float(2) # needs to be fixed to be dynamic
          else:            
            dframe.loc[dframe.fish_id == current_fish_a_id, "fish_id"] = float(2) # needs to be fixed to be dynamic
            dframe.loc[dframe.fish_id == current_fish_b_id, "fish_id"] = float(1) # needs to be fixed to be dynamic
            
          
                  
                                
          '''if (similarity_acurrent_aaverage > similarity_acurrent_baverage):
            dframe.loc[dframe.fish_id == current_fish_a_id, "fish_id"] = float(1) # needs to be fixed to be dynamic
            dframe.loc[dframe.fish_id == current_fish_b_id, "fish_id"] = float(2) # needs to be fixed to be dynamic              
            make_copy_last_seen = True
            print('1')
          else:
            dframe.loc[dframe.fish_id == current_fish_a_id, "fish_id"] = float(2) # needs to be fixed to be dynamic
            dframe.loc[dframe.fish_id == current_fish_b_id, "fish_id"] = float(1) # needs to be fixed to be dynamic              
            make_copy_last_seen = True
            print('2')
                
            else:
              
              print('2222')
              
              if similarity_bcurrent_aaverage > similarity_bcurrent_baverage:
                dframe.loc[dframe.fish_id == current_fish_a_id, "fish_id"] = float(2) # needs to be fixed to be dynamic
                dframe.loc[dframe.fish_id == current_fish_b_id, "fish_id"] = float(1) # needs to be fixed to be dynamic              
                make_copy_last_seen = True 
                print('1')                            
              else:
                print('2')
                dframe.loc[dframe.fish_id == current_fish_a_id, "fish_id"] = float(1) # needs to be fixed to be dynamic
                dframe.loc[dframe.fish_id == current_fish_b_id, "fish_id"] = float(2) # needs to be fixed to be dynamic              
                make_copy_last_seen = True
          
          #else:
            #update_counter -= 1'''
                  
                          
              
          
        if update_counter >= 2:
          previous_histograms = histograms.copy()    
    

            
                        
           
    update_counter += 1
    if update_counter > 100:
      update_counter = 100

    
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


