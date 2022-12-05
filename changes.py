#usage:
# python script.py [path to video with filename] [path to save without file name] [path to metadata with filename]

debug = True
trait_to_analyze = "color" # ["color", "length"]

import sys

#print('path_to_video: ', sys.argv[1])
#print('save_results_on: ', sys.argv[2])
#print("path_to_metadata: ", sys.argv[3])
from IPython.display import clear_output
import pandas as pd
if debug==True:
  import winsound

path_to_video = "C:/Users/marcio/Videos/Ian_videos/croped_Ian/errors/20191115_1615_11-1_R_B.avi" #sys.argv[1]
path_to_save = "C:/Users/marcio/Videos/Ian_videos/croped_Ian/errors/" #sys.argv[2]
path_to_meta = "C:/Users/marcio/Videos/Ian_videos/MIKK_F0_metadata.csv" # sys.argv[3]
import os
import pathlib

final_path = pathlib.PurePath(path_to_video)
expe = final_path.name

df_meta = pd.read_csv(path_to_meta)

initial_frame = 3600 #df_meta.loc[df_meta["sample"] == expe[:-4]]["of_start"].iloc[0] + 200
final_frame = df_meta.loc[df_meta["sample"] == expe[:-4]]["of_end"].iloc[0]

if os.path.exists(path_to_save + "/" + expe[:-4] + '.csv'):
  os.remove(path_to_save + "/" + expe[:-4] + '.csv')
  print("CSV file exist, It has been removed to a new one be created")
else:
  print("CSV file does not exist, it will be created")

 
with open(path_to_save + "/" + expe[:-4] + '.csv', 'w') as fd:
  fd.write('frame_number,length_of_fish,center_of_mass,fish_tail,fish_head,quadrant,fish_area,fish_id,tail_points,quad_coord,sum_chanel_B,sum_chanel_G,sum_chanel_R,avg_chanel_B,avg_chanel_G,avg_chanel_R,count_chanel\n')

from collections import deque
import cv2
import scipy.stats as stats
import scikit_posthocs as sp
from scipy.stats import kurtosis
import scipy.stats
import numpy as np
import math
import matplotlib.pyplot as plt 
from skimage.morphology import skeletonize, thin



def crop_img(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # invert gray image
  gray = 255 - gray

    # gaussian blur
  blur = cv2.GaussianBlur(gray, (3,3), 0)

    # threshold
  thresh = cv2.threshold(blur,236,255,cv2.THRESH_BINARY)[1]

    # apply close and open morphology to fill tiny black and white holes
  kernel = np.ones((5,5), np.uint8)
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # invert thresh
  thresh = 255 -thresh

    # get contours (presumably just one around the nonzero pixels) 
    # then crop it to bounding rectangle
  contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1]
  cntr = contours[0]
  x,y,w,h = cv2.boundingRect(cntr)
  return img[y:y+h, x+1:x+w-1]   # the 1 is to correct for a small black strip on images



# Check if camera opened successfully
cap = cv2.VideoCapture(path_to_video)


if final_frame == None:
  final_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
#generate the background dynamically:
background_frame = []
step = (int((final_frame-initial_frame)/20))
for idx_frame in range(initial_frame,final_frame,step):
  cap.set(1, idx_frame)  
  # Capture frame-by-frame
  ret, frame = cap.read()

  frame = crop_img(frame)    # crop black borders
  

  if ret == True:   
    background_frame.append(frame)

    
#result = scipy.stats.mode(np.stack(background_frame), axis=0)
result = np.median(background_frame,axis=0)
dynamic_background = result.astype(np.uint8)
#dynamic_background = result.mode[0]

#cv2.imshow('dyn',dynamic_background)




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

 
#pd.set_option('display.max_columns', None) 
#background_img = cv2.imread(dynamic_background)
bw_back = cv2.bilateralFilter(dynamic_background,9,75,75)

bw_back = cv2.cvtColor(bw_back, cv2.COLOR_BGR2GRAY)
#bw_back = cv2.GaussianBlur(bw_back, (9,9) ,0)
#the_plot = result = bw_back.flatten()
#plt.hist(the_plot, bins="auto") 
#plt.show()

the_median = np.median(bw_back)    
bw_back[bw_back < the_median -40] = 0


 

ret, mask_fillin = cv2.threshold(bw_back,0,255,cv2.THRESH_BINARY_INV)

kernel = np.ones((5, 5), np.uint8)
mask_fillin = cv2.dilate(mask_fillin, kernel, iterations=3)


#cv2.imshow("mask_fillin", mask_fillin)

bw_back = cv2.inpaint(bw_back, mask_fillin, 10, cv2.INPAINT_TELEA)



cv2.imshow("paint", bw_back)


# find the quadrants
blur = cv2.medianBlur(bw_back, 9)
thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
      cv2.THRESH_BINARY,27,2)

cv2.imshow("thresh", thresh)


kernel = np.ones((5,5),np.uint8)
#opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
erosion = cv2.erode(thresh,kernel,iterations = 1)

cv2.imshow("dilation", erosion)
cnts = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

max_area = (frame.shape[0] * frame.shape[1])/4
min_area = max_area * 0.5
quadrants_lines = []
quadrant_values = []
for c in cnts: 
    area = cv2.contourArea(c)          
    if area > min_area and area < max_area:
        cv2.drawContours(frame, c, -1, (0, 255, 0), 3)
        cv2.imshow("quad", frame)
        x,y,w,h = cv2.boundingRect(c)        
        quadrants_lines.append((x,y,w,h))
if len(quadrants_lines) != 4:
  print("Number od detected quadrants is not 4")
  quit()        
else:  
  print("4 quadrants found")




   
#bw_back = cv2.fastNlMeansDenoising(bw_back, None, 15.0, 15, 29)

############################


#avging = cv2.blur(bw_back,(100,100))
   

#bw_back[bw_back<50] = avging[bw_back<50]




#the_median = int(np.median(bw_back))
#bw_back[bw_back < 256] = the_median

#create a blank image to plot everything on
blank_image = np.zeros((bw_back.shape[0], bw_back.shape[1], 3), np.uint8)

previous_df = None
lopp = 0

previous_id_fish_local = []

histograms_ids = {'10':deque(maxlen=60), '11':deque(maxlen=60), '12':deque(maxlen=60), '13':deque(maxlen=60), '20':deque(maxlen=60), '21':deque(maxlen=60), '22':deque(maxlen=60), '23':deque(maxlen=60)}
histograms_X_Y = {"X0":deque(maxlen=60), "X1":deque(maxlen=60), "X2":deque(maxlen=60), "X3":deque(maxlen=60), "Y0":deque(maxlen=60), "Y1":deque(maxlen=60), "Y2":deque(maxlen=60), "Y3":deque(maxlen=60)}

if debug==True:
  cv2.imshow('bw_back' , bw_back)

  cv2.waitKey(1)


if (cap.isOpened()== False): 
  print("Error opening video stream or file")

for idx, coordinates in enumerate(quadrants_lines):
      x,y,w,h = coordinates
      cv2.rectangle(frame, (x, y), (x + w, y + h), (36,255,12), 2)
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(frame, str(idx),(x, y+25), font, 1, (255,255,255), 2) 

#frame_t = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#plt.imshow(frame_t)        
#plt.gcf().set_size_inches(13, 12)
#plt.title(idx_frame)
#plt.gcf().set_size_inches(23, 22)
#plt.show()    
    
    #cv2.imshow('Main',frame)  ####### here    

#clear_output(wait=True)



               
for idx_frame in range(0, initial_frame, 1):   # only to check wich fish is icab
  print(idx_frame)
    
  
  cap.set(1, idx_frame)
  
  # Capture frame-by-frame
  ret, frame = cap.read()

  frame = crop_img(frame)
  if ret == True:
      if debug==True:
        cv2.imshow('Main',frame)
        cv2.waitKey(1)  
      
      
      
      
      
      
        
        ###########################################################
for idx_frame in range(initial_frame,final_frame,1):   #3000 to 4000 
  print(idx_frame)
    
  
  cap.set(1, idx_frame)
  
  # Capture frame-by-frame
  ret, frame = cap.read()

  frame = crop_img(frame)
  if ret == True:    
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
        
    #bw_mainImage = cv2.bilateralFilter(frame,9,75,75) # cv2.GaussianBlur(bw_mainImage, (9,9) ,0)

    bw_mainImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow("sharpened", bw_mainImage)
    
    #an_array = np.full(bw_mainImage.shape, 255, dtype=np.uint8)
    #back_inv = an_array - bw_back
    
    if debug==True:
      cv2.imshow("bw_back", bw_back)
    
    #cv2.imwrite('C:/Users/marcio/Pictures/background/back_gd.jpg', bw_back)

    
    #bw_mainImage = cv2.fastNlMeansDenoising(bw_mainImage, None, 3.0, 7, 21)
    
    #cv2.imshow("bw_mainImagexxx", bw_mainImage)

    
    
    #kernel = np.array([[0, -1, 0],
                   #[-1, 5,-1],
                   #[0, -1, 0]])
    #bw_mainImage = cv2.filter2D(src=bw_mainImage, ddepth=-1, kernel=kernel)
    #bw_back = cv2.filter2D(src=bw_back, ddepth=-1, kernel=kernel) 


    bw_mainImage = bw_mainImage.astype(float)
    
    diff = bw_mainImage - bw_back #cv2.absdiff(bw_back, bw_mainImage)
    diff[diff > 0] = 0
    diff = np.absolute(diff)
    #diff[diff == 255] = 0
    diff = diff.astype(np.uint8)

    #plt.imshow(diff)        
    #plt.gcf().set_size_inches(13, 12)
    #plt.title(idx_frame)
    #plt.show()
    
 
    
    
    
      
    edges = cv2.Canny(diff,10,100)
    
    kernel = np.ones((5,5),np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

      
    cannymerged = np.maximum(diff, edges)
    
    if debug==True:
      cv2.imshow("cannymerged", cannymerged)      
      cv2.imshow("diff", diff)      
      cv2.imshow("canny", edges)
     
          
    ret,thresh = cv2.threshold(cannymerged,15,255,cv2.THRESH_BINARY) #15
    
    kernel = np.ones((2,2),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


   

    if debug==True:
      cv2.imshow("thress", thresh)
    
    
    
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
    skeleton_list = []
    quad_coord = []
    list_sum_chanel_B = []
    list_sum_chanel_G = []
    list_sum_chanel_R = []
    list_avg_chanel_B = [] 
    list_avg_chanel_G = []
    list_avg_chanel_R = []
    list_count_chanel = []
    list_red_i = []

    counter = 0
    
    for idx, cnt in enumerate(contours):
                  
      area = cv2.contourArea(cnt) 
      #calculate aspect ratio for template quality filtering   

      if area > 150 and area < 2000:
        
        mask = np.zeros(frame.shape, dtype=np.uint8)
        mask = cv2.drawContours(mask, [cnt], -1, color=(255,255,255),thickness=-1)          
        
        
        
        croped_fish = cv2.bitwise_and(frame, mask)
        #cv2.imshow('croped', croped_fish)
        test_hist = croped_fish.copy()
        croped_fish = croped_fish.astype('float')
        croped_fish[croped_fish == 0] = np.nan
        
        #cv2.imshow('hist', result1)
        sum_chanel_B = np.nansum(croped_fish[:,:,0]) 
        sum_chanel_G = np.nansum(croped_fish[:,:,1]) 
        sum_chanel_R = np.nansum(croped_fish[:,:,2]) 
        avg_chanel_B = np.nanmean(croped_fish[:,:,0]) 
        avg_chanel_G = np.nanmean(croped_fish[:,:,1]) 
        avg_chanel_R = np.nanmean(croped_fish[:,:,2])
        count_chanel = np.count_nonzero(~np.isnan(croped_fish[:,:,0]))
        red_i = (sum_chanel_R/sum_chanel_B)
        #plt.hist(croped_fish[:,:,0])
        #plt.show()
        
        
        ###############################
        """img = cv2.cvtColor(test_hist, cv2.COLOR_BGR2RGB)
    

        red_hist = cv2.calcHist([img], [0], None, [255], [1, 255])
        green_hist = cv2.calcHist([img], [1], None, [255], [1, 255])
        blue_hist = cv2.calcHist([img], [2], None, [255], [1, 255])
      
        plt.subplot(4, 1, 1)
        plt.imshow(img)
        plt.title('image')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(4, 1, 2)
        plt.plot(red_hist, color='r')
        plt.xlim([0, 255])
        plt.title('red histogram')

        plt.subplot(4, 1, 3)
        plt.plot(green_hist, color='g')
        plt.xlim([0, 255])
        plt.title('green histogram')

        plt.subplot(4, 1, 4)
        plt.plot(blue_hist, color='b')
        plt.xlim([0, 255])
        plt.title('blue histogram')

        plt.tight_layout()
        plt.show()"""

##########   
        
        
        #will be used for squeleton
        drawn_image_for_skeleton = blank_image.copy()
        drawn_image_for_skeleton = cv2.drawContours(drawn_image_for_skeleton, [cnt], -1, color=(255,255,255),thickness=-1)
        
        
        
        bw_mainImage_sk = cv2.cvtColor(drawn_image_for_skeleton, cv2.COLOR_BGR2GRAY)
        binarizedImage = bw_mainImage_sk / 255
        skeleton = skeletonize(binarizedImage)
        skeleton = (skeleton*255).astype(np.uint8)
        #cv2.imshow("squeleton", skeleton)  
      
        skeleton_coords = list(zip(*np.nonzero(skeleton)))
              
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
        tail_coords = list_of_points[max_index_tail]
        
        
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
        
        lenght_of_fish = math.sqrt( (aver_head[0] - tail_coords[0])  **2 + (aver_head[1] - tail_coords[1])**2 )
        
        
        #squeleton continuity       
        distances_skeleton_from_head = []
        for l in skeleton_coords:           
          distance = math.sqrt(   (aver_head[0]-l[1])**2 + (aver_head[1]-l[0])**2    )
          distances_skeleton_from_head.append(distance)        
        sorted_index = np.argsort(np.array(distances_skeleton_from_head))
                  
        lenght = len(skeleton_coords)      
        step = int(lenght*.70/3)
        if step == 0:
          step=1        
        tail_points_filtered = []
        for x in range(lenght-1, 0, -step):
          tail_points_filtered.append(skeleton_coords[sorted_index[x]])     #need to reverse tuple of squeleton as it is y,x not x,y  
               
        tail_points_filtered.reverse()

        
        
        
        
        


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
        
        ######print quad
  
        
        
        
        ver_line = max(list(zip(*quadrants_lines))[0]) - 20
        hor_line = max(list(zip(*quadrants_lines))[1]) - 20
        
        cv2.circle(frame, (fish_COM[0], fish_COM[1]), 2, (255, 0, 0), -1)
        cv2.circle(frame, (int(ver_line), int(hor_line)), 2, (255, 0, 0), -1)
        
        if (fish_COM[0] < ver_line)  and (fish_COM[1] > hor_line):  
          quadrant_value = 0
        elif (fish_COM[0] > (hor_line) and fish_COM[1] > (ver_line)):
          quadrant_value = 1
        elif (fish_COM[0] > (ver_line) and fish_COM[1] < (hor_line)):
          quadrant_value = 2
        elif (fish_COM[0] < (ver_line) and fish_COM[1] < (hor_line)):
          quadrant_value = 3
        else:
          print("None of quadrant fits!!!")
          quit()
          #quadrant_value = 0
      

        #store variables locally in the loop for immediate calculation purposes
        idx_local.append(counter)          
        lenght_of_fish_local.append(lenght_of_fish)
        position_fish_local.append(fish_COM)
        fish_tail_local.append(tail_coords)
        fish_head_local.append(aver_head)
        quadrant_local.append(quadrant_value)
        fish_area.append(area)        
        countours_idx.append(0)
        template_area.append(area_rec)
        template_blur.append(0)
        template_dark.append(0)
        skeleton_list.append(tail_points_filtered) 
        quad_coord.append(quadrants_lines[quadrant_value])
        list_sum_chanel_B.append(sum_chanel_B)
        list_sum_chanel_G.append(sum_chanel_G)
        list_sum_chanel_R.append(sum_chanel_R)
        list_avg_chanel_B.append(avg_chanel_B)
        list_avg_chanel_G.append(avg_chanel_G)
        list_avg_chanel_R.append(avg_chanel_R)
        list_count_chanel.append(count_chanel)      
        list_red_i.append(red_i)
        if trait_to_analyze == "length":
          histograms.append(fish_pectoral_lenght)
        elif trait_to_analyze == "color":
          histograms.append(red_i)
        else:
          print("need specify the trait to compare")
          quit()
                        
        counter +=1
    
    
    dframe = pd.DataFrame(idx_local)
    dframe['lenght_of_fish_local'] = lenght_of_fish_local
    dframe['position_fish_local'] = position_fish_local
    dframe['fish_tail_local'] = fish_tail_local
    dframe['fish_head_local'] = fish_head_local
    dframe['quadrant_local'] = quadrant_local
    dframe['fish_area'] = fish_area
    dframe['cnt_idx'] = countours_idx
    dframe["fish_id"] = None
    dframe["tail_points"] = skeleton_list
    dframe["quad_coord"] = quad_coord    
    dframe["sum_chanel_B"] = list_sum_chanel_B  
    dframe['sum_chanel_G'] = list_sum_chanel_G
    dframe['sum_chanel_R'] = list_sum_chanel_R
    dframe["avg_chanel_B"] = list_avg_chanel_B
    dframe["avg_chanel_G"] = list_avg_chanel_G
    dframe["avg_chanel_R"] = list_avg_chanel_R
    dframe["count_chanel"] = list_count_chanel
    dframe["red_i"] = list_red_i   
    #print('test')
   
   
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
          dframe.loc[(dframe.fish_id == previous_fish_1_id) & (dframe.quadrant_local == row_q), "fish_id"] = 1
          histograms_ids['2' + str(row_q)] = histograms_X_Y[str(previous_fish_2_id) + str(row_q)]          
          dframe.loc[(dframe.fish_id == previous_fish_2_id) & (dframe.quadrant_local == row_q), "fish_id"] = 2          
        print("time to decide in frame ", idx_frame)
        # frequency is set to 500Hz
        freq = 500        
        # duration is set to 100 milliseconds            
        dur = 1000                      
        if debug==True:
          winsound.Beep(freq, dur)    
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
        

        df1 = pd.DataFrame({'score': templates_of_ID1,
                  'group': 'id1'}) 
        
   
        df2 = pd.DataFrame({'score': templates_of_ID2,
                  'group': 'id2'}) 
        #
        df3 = pd.DataFrame({'score': X_current_template_list,
                  'group': 'X'})     

        df4 = pd.DataFrame({'score': Y_current_template_list,
                  'group': 'Y'}) 
        
        df = pd.concat([df1, df2, df3, df4])     
        
        tukey = sp.posthoc_ttest(df, val_col='score', group_col='group', p_adjust='holm')
        
        
        X1 = tukey['X'][0]
        X2 = tukey['X'][1]
        Y1 = tukey['Y'][0]
        Y2 = tukey['Y'][1]         
      
      
        minimum =  min(X1, X2, Y1, Y2)       
              
        if minimum == X2 or minimum == Y1:
          dframe.loc[(dframe.fish_id == 'X') & (dframe.quadrant_local == row_q), "fish_id"] = 1
          dframe.loc[(dframe.fish_id == 'Y') & (dframe.quadrant_local == row_q), "fish_id"] = 2
        else:
          dframe.loc[(dframe.fish_id == 'X') & (dframe.quadrant_local == row_q), "fish_id"] = 2
          dframe.loc[(dframe.fish_id == 'Y') & (dframe.quadrant_local == row_q), "fish_id"] = 1
        
        pass            

      #####################################################################################   
    #block 6
    # Time to draw everything on a template    
        
    for c in idx_local[:8]:     
      #head
      cv2.circle(frame, fish_head_local[c], 2, (226, 43, 128), -1)
      #center of mass
      #cv2.circle(frame, position_fish_local[c], 2, (0, 165, 255), -1)      

    position_list = dframe.position_fish_local.tolist()
    fish_id = dframe.fish_id.tolist()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for indice, value in enumerate(fish_id):
     
      
      cv2.putText(frame, 'id: ' + str(fish_id[indice]) + " - " + 'qu: ' + str(quadrant_local[indice]),(position_list[indice]), font, 0.4,(0,0,0),1)
     
    
      
    
    #for c in idx_local[:8]:
      #for coords in skeleton_list[c]:      
        #cv2.circle(frame, (coords[1], coords[0]), 2, (0, 0, 255), -1)
        
    

    
    
    #write the quadrants
    for idx, coordinates in enumerate(quadrants_lines):
      x,y,w,h = coordinates
      cv2.rectangle(frame, (x, y), (x + w, y + h), (36,255,12), 2)
      font = cv2.FONT_HERSHEY_SIMPLEX
      #cv2.putText(frame, str(idx),(x, y+25), font, 1, (255,255,255), 2) 

   
    
    #select fields to write to csv
    filt_dframe = dframe.loc[(dframe['fish_id'] == "X") | (dframe['fish_id'] == "Y") | (dframe['fish_id'] == 1) | (dframe['fish_id'] == 2)]
    filt_s_dataf = filt_dframe[['lenght_of_fish_local', 'position_fish_local', 'fish_tail_local', 'fish_head_local', 'quadrant_local', 'fish_area', 'fish_id', 'tail_points', 'quad_coord', "sum_chanel_B", "sum_chanel_G", 'sum_chanel_R', "avg_chanel_B", "avg_chanel_G", "avg_chanel_R", "count_chanel"]]
    filt_s_dataf.insert(loc=0,
          column='frame_number',
          value=idx_frame)
    
    filt_s_dataf.to_csv(path_to_save + "/" + expe + '.csv', mode='a', index=False, header=False)
   

    if debug==True:
      cv2.imshow('Main',frame)  ####### here    
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #plt.imshow(frame)        
    #plt.gcf().set_size_inches(13, 12)
    #plt.title(idx_frame)
    #plt.show()
    #clear_output(wait=True)
    
    
    
    
    previous_df = dframe.copy()  
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break



# When everything done, release the video capture object
cap.release()

# Closes all the frames
#cv2.destroy