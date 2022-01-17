quadr = 'B'
import cv2
import statistics 
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

histograms_ids = {1:[], 2:[]}

#cv2.imshow('background' , bw_back)

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
for idx_frame in range(7870,10000000,1):   #3000 to 4000
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
        
        #the head coordinates
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
          if distances_cm[e] < 20:
            list_value_cm = cnt[e].tolist()              
            nearest_values_for_template.append(list_value_cm)

        arr_for_template = np.array(nearest_values_for_template)
        
        contours_body = arr_for_template
        
        # determine the most extreme points along the contour of the tail cropped fish
        extLeft = tuple(contours_body[contours_body[:, :, 0].argmin()][0])
        extRight = tuple(contours_body[contours_body[:, :, 0].argmax()][0])
        extTop = tuple(contours_body[contours_body[:, :, 1].argmin()][0])
        extBot = tuple(contours_body[contours_body[:, :, 1].argmax()][0])
        #list_of_points = [extLeft, extRight, extTop, extBot]
        #print(extLeft)
        copy_frame = original_img.copy()
        copy_frame[:extTop[1], :, :] = 255
        copy_frame[extBot[1]:, :, :] = 255
        copy_frame[:, :extLeft[0], :] = 255
        copy_frame[:, extRight[0]:, :] = 255
        
      
        
        #calculate the angle and rotate the black masked image
        aver_cm = np.mean(arr, axis=0).astype(int)
        
        #fish_center of mass
        fish_COM = (aver_cm[0][0], aver_cm[0][1])
        
        
        #create a fake y axis, as in images, the y is inverted
    
        
        inverted_aver_head = (aver_head[0], frame.shape[0] - aver_head[1]) 
        inverted_fish_COM = (fish_COM[0], frame.shape[0] - fish_COM[1])

                
        p1 = inverted_aver_head
        p2 = inverted_fish_COM
        
        #print(inverted_aver_head)
        #print(inverted_fish_COM)
                # Difference in x coordinates
        myradians = math.atan2(p1[0]-p2[0], p1[1]-p2[1])
        mydegrees = math.degrees(myradians)
        
        #print("degre")
        #print(mydegrees)
        
        
        from PIL import Image
        im_pil = Image.fromarray(copy_frame)
        #rotated = im_pil.rotate(, expand=True)
        rotated =  im_pil.rotate(mydegrees, fillcolor=tuple(np.mean(np.array(im_pil)[0,:], axis=0).astype(int)), expand=True)

        rotated_template = np.asarray(rotated)
              
       
        
        #rotated_template = im_pil.rotate(-mydegrees, fillcolor=tuple(np.mean(np.array(im_pil)[0,:], axis=0).astype(int)))

        
        bw_back_rt = cv2.cvtColor(rotated_template, cv2.COLOR_BGR2GRAY)
        
   
        ret,thresh = cv2.threshold(bw_back_rt,170, 255,cv2.THRESH_BINARY)
        
        
        #cv2.imshow('error', thresh)
        #cv2.waitKey(0)
        
        thresh = thresh.clip(max=1)
        
        inverted =1-thresh
        
       
        
        #coords = np.argwhere(thresh)
        mask2 = np.argwhere(inverted)
        
      
        
        x0, y0 = mask2.min(axis=0)
        x1, y1 = mask2.max(axis=0) + 1   # slices are exclusive at the top
        
        
        
                
        #before apply the cut coordinates, grab a rotated main image
        im_pil = Image.fromarray(original_img.copy())
        rotated_original =  im_pil.rotate(mydegrees, fillcolor=tuple(np.mean(np.array(im_pil)[0,:], axis=0).astype(int)), expand=True)
        rotated_original = np.asarray(rotated_original)

        
        final_template = rotated_original[x0:x1, y0:y1, :]

        
        
        
        
        
        
        
        
        
                

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
        fish_area.append(area)
        histograms.append(final_template)   # need to fix afterwards
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
    
    #qprint(histograms_ids)
    if previous_df is None:
      
      previous_df = dframe.copy()
      previous_histograms = histograms.copy()
      previous_histograms_ids = histograms_ids.copy()     
     
      previous_df.loc[previous_df['quadrant_local'] == quadr, 'fish_id'] = [x for x in [1, 2]]
      list_idx = previous_df.loc[previous_df['quadrant_local'] == quadr].index.tolist()
         
      previous_histograms_ids[1].append(histograms[list_idx[0]])
      previous_histograms_ids[2].append(histograms[list_idx[1]])

       
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
          histograms_X_Y = {"X":[], "Y":[]}
                
          continue

        
             
        if update_counter == 1:                    
          dframe.loc[dframe['quadrant_local'] == quadr, 'fish_id'] = [x for x in ['X', 'Y']]          
          list_idx = dframe.loc[dframe['quadrant_local'] == quadr].index.tolist()          
          histograms_X_Y['X'].append(histograms[list_idx[0]])
          histograms_X_Y['Y'].append(histograms[list_idx[1]])
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
            if update_counter  <= 20:  
              histograms_X_Y[previous_fish_1_id].append(histograms[row['original_index']]) #((np.add((np.multiply(previous_histograms_X_Y[previous_fish_1_id], 3)), histograms[row['original_index']]) / 4)).astype(np.float32)
            else:
              histograms_ids[int(previous_fish_1_id)].append(histograms[row['original_index']]) #((np.add((np.multiply(previous_histograms_ids[int(previous_fish_1_id)], 3)), histograms[row['original_index']]) / 4)).astype(np.float32)
              if len(histograms_ids[int(previous_fish_1_id)]) > 20:
                #del histograms_ids[int(previous_fish_1_id)] = histograms_ids[int(previous_fish_1_id)][-20:]
                del histograms_ids[int(previous_fish_1_id)][0]
              print(len(histograms_ids[int(previous_fish_1_id)]))
              
              #cv2.imshow('isso é um', histograms_ids[int(previous_fish_1_id)][-1])
  
              #cv2.waitKey(0)  
              #print(histograms_ids[2])
              #cv2.waitKey(0)
              
          else: #the same as above, but in a inverse way
            dframe.loc[row['original_index'],'fish_id'] = previous_fish_2_id
            dframe.loc[row['original_index'],'fish_area'] = (previous_fish_2_area * 40 + row['fish_area'])/21
            
            #update the histograms
            if update_counter  <= 20:  
              histograms_X_Y[previous_fish_2_id].append(histograms[row['original_index']]) #((np.add((np.multiply(previous_histograms_X_Y[previous_fish_2_id], 3)), histograms[row['original_index']]) / 4)).astype(np.float32)
              
            else:
              histograms_ids[int(previous_fish_2_id)].append(histograms[row['original_index']]) #((np.add((np.multiply(previous_histograms_ids[int(previous_fish_2_id)], 3)), histograms[row['original_index']]) / 4)).astype(np.float32)
              if len(histograms_ids[int(previous_fish_2_id)]) > 20:
                del histograms_ids[int(previous_fish_2_id)][0] #=histograms_ids[int(previous_fish_2_id)][-20:]
              print(len(histograms_ids[int(previous_fish_2_id)]))
              
              #cv2.imshow('isso é dois', histograms_ids[int(previous_fish_2_id)][-1])  
              #cv2.waitKey(0)
              
        if update_counter > 0 and update_counter < 20:
          previous_histograms_X_Y = histograms_X_Y.copy()  
          
          
        
        # here we decide which fish is 1 and 2 based on X and Y
        if update_counter == 20:
          
          current_fish = dframe.loc[dframe['quadrant_local'] == row_q]
          current_fish_a = current_fish.iloc[0]
          current_fish_b = current_fish.iloc[1]
          
          # recover the lists of templates to hash
          
          
          current_fish_a_id = current_fish_a['fish_id']
          hist_of_a = histograms_X_Y[current_fish_a_id]
           
          current_fish_b_id = current_fish_b['fish_id']
          hist_of_b = histograms_X_Y[current_fish_b_id]
                  
          a_histo_last_seen = histograms_X_Y['X']    
          
          b_histo_last_seen = histograms_X_Y['Y'] 
          
            
          ########### hash images ##########
          main_hash = []
          for idx_current, x in enumerate(current_fish_a_id):
            for idx_last_seen, y in enumerate(a_histo_last_seen):
              
              hash_list = []
                                                
              im_pil = Image.fromarray(a_histo_last_seen[idx_last_seen])
              hash1 = imagehash.whash(im_pil)
              print(hash1)
              cv2.imshow('a_histo_last_seen',a_histo_last_seen[idx_last_seen])                         
                    
              im_pil = Image.fromarray(b_histo_last_seen[idx_last_seen])
              hash2 = imagehash.whash(im_pil)
              print(hash2)
              cv2.imshow('b_histo_last_seen', b_histo_last_seen[idx_last_seen])                     
              
              im_pil = Image.fromarray(hist_of_a[idx_current])
              hash3 = imagehash.whash(im_pil)
              print(hash3)            
              cv2.imshow('hist_of_a', hist_of_a[idx_current])         
              
              im_pil = Image.fromarray(hist_of_b[idx_current])
              hash4 = imagehash.whash(im_pil) # p = 12,  crop, color, whash(im_pil, mode='db4')
              print(hash4)            
              cv2.imshow('hist_of_b', hist_of_b[idx_current])             
              
              hash_list.append(hash1-hash3)
              hash_list.append(hash1-hash4)
              hash_list.append(hash2-hash3)
              hash_list.append(hash2-hash4)
              
              main_hash.append(hash_list)
          
          list_of_first_comp = []
          list_of_second_comp = []
          list_of_third_comp = []
          list_of_fourth_comp =[]
          
          for set_of_hashs in main_hash:
            list_of_first_comp.append(set_of_hashs[0])
            list_of_second_comp.append(set_of_hashs[1])
            list_of_third_comp.append(set_of_hashs[2])
            list_of_fourth_comp.append(set_of_hashs[3])
                
          average_first_comp = sum(list_of_first_comp)
          average_second_comp = sum(list_of_second_comp)
          average_third_comp = sum(list_of_third_comp)
          average_fourth_comp = sum(list_of_fourth_comp)
          
          print(average_first_comp)
          print(average_second_comp)
          print(average_third_comp)
          print(average_fourth_comp)
          
          cv2.waitKey(0)
          
          min_value = min([average_first_comp, average_second_comp, average_third_comp, average_fourth_comp])
          
          if min_value == average_first_comp or min_value == average_fourth_comp:
            dframe.loc[dframe.fish_id == current_fish_a_id, "fish_id"] = float(2) 
            dframe.loc[dframe.fish_id == current_fish_b_id, "fish_id"] = float(1) 
          else:
            dframe.loc[dframe.fish_id == current_fish_a_id, "fish_id"] = float(1) 
            dframe.loc[dframe.fish_id == current_fish_b_id, "fish_id"] = float(2) 
              #cv2.waitKey(0)
              
              #cv2.destroyAllWindows() 
          
          #update_counter = update_counter -1
       
          
          
         

         
    
          """if max_simil == similarity_acurrent_aaverage or max_simil == similarity_bcurrent_baverage:
            dframe.loc[dframe.fish_id == current_fish_a_id, "fish_id"] = float(1) # needs to be fixed to be dynamic
            dframe.loc[dframe.fish_id == current_fish_b_id, "fish_id"] = float(2) # needs to be fixed to be dynamic
          else:            
            dframe.loc[dframe.fish_id == current_fish_a_id, "fish_id"] = float(2) # needs to be fixed to be dynamic
            dframe.loc[dframe.fish_id == current_fish_b_id, "fish_id"] = float(1) # needs to be fixed to be dynamic"""
            
          
                  
                                
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
                  
                          
              
          
        if update_counter >= 20:
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


