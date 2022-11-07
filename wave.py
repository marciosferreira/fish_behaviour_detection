

# Check if camera opened successfully
import pickle

import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
import math


with open('C:/Users/marcio/Documents/results_Ian/dict.txt', 'rb') as handle:
  frames_numbers = pickle.loads(handle.read())


df = pd.DataFrame.from_dict(frames_numbers, orient='index')
df.columns=["fish_head_x", "fish_head_y", "sequence", "tail_coords"]
df["angle_corr_tail"] = None
print(df)



plt.ion()


cap = cv2.VideoCapture('C:/Users/marcio/Videos/Ian_videos/20191121_1454_iCab_L_C.avi')
final_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

from itertools import cycle

color_cycle = cycle(((0,0,255),(0,255,0),(255,0,0)))

#frame_n = 0

previous_id = 0
   
# Read until video is completed
for idx_frame in range(0,final_frame,1):  
  if idx_frame in frames_numbers:  
    print(idx_frame)

      
    
    cap.set(1, idx_frame)
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
      # Display the resulting frame
      
     
      
          if previous_id != frames_numbers[idx_frame][2]:
            color = next(color_cycle)
          cv2.rectangle(frame, (frames_numbers[idx_frame][0]-30,frames_numbers[idx_frame][1]-30), (frames_numbers[idx_frame][0]+30, frames_numbers[idx_frame][1]+30), color, 2)
          font = cv2.FONT_HERSHEY_SIMPLEX
          cv2.putText(frame, str(frames_numbers[idx_frame][2]),(frames_numbers[idx_frame][0]-30,frames_numbers[idx_frame][1]-30), font, 0.5, (0,0,0), 1) 
          previous_id = frames_numbers[idx_frame][2]
          
          
          for coords in frames_numbers[idx_frame][-1]:      
          #tail plot
          #cv2.circle(drawn_image, fish_tail_local[c], 2, (0, 0, 255), -1)
            cv2.circle(frame, (coords[1], coords[0]), 2, (0, 0, 255), -1)
            
    

          
          
          cv2.imshow('Frame', frame)
          
          cv2.waitKey(300)
          
  

          if 1 == 1:
            # Define x, y, and xnew to resample at.      
            x_list = list(zip(*frames_numbers[idx_frame][3]))[1]
            #x_final = [float(i)/min(x_list) for i in x_list]


            
            #print(y)
            y_list_inv = list(zip(*frames_numbers[idx_frame][3]))[0]
            y_list = [i for i in y_list_inv] # invert the y axis 870
            
            #y_final = [float(i)/min(y_list) for i in y_list]
            
            
            tuple_y_inv = tuple(zip(x_list, y_list))
            
           
            import math

            def calc_rotation(coords_body):
              x = coords_body[0][0] - coords_body[1][0]             
              y = coords_body[0][1] - coords_body[1][1]              
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
          
            
           
              
            corrected_angle = [rotate(i, math.radians(angle), tuple_y_inv[0]) for i in tuple_y_inv]
            
       
     
        
            

            x_list_2 = list(zip(*corrected_angle))[0]
            y_list_2 = list(zip(*corrected_angle))[1]
            
            x_norm = x_list_2 #[i-x_list_2[0] for i in x_list_2]
            y_norm = y_list_2 #[i-y_list_2[0] for i in y_list_2]
            
            
            
            x_norm_2 = [float(i)/x_norm[0] for i in x_norm]
            y_norm_2 = [float(i)/y_norm[0] for i in y_norm]
            
            
            
            #corrected_tails_points = tuple(zip(x_norm_2, y_norm_2))
            corrected_tails_points = [list(a) for a in zip(x_norm_2, y_norm_2)]
            #insert normalized to df
            df.at[idx_frame, "angle_corr_tail"] = corrected_tails_points        # frames_numbers[idx_frame]
            
            
         
            
            #xnew = np.linspace(x_norm_2[0], x_norm_2[-1])           
     
            #f_cubic = interp1d(x_norm_2, y_norm_2, kind='quadratic')
          
            #plt.plot(x_norm_2, y_norm_2, 'o', label='data')
            #plt.draw()
            #plt.plot(x_norm_2, y_norm_2, 'o', label='data')
            #plt.draw()
            #plt.plot(xnew, f_linear(xnew), '-', label='linear')
            #plt.xlim(0.1925, 0.21)
            #plt.ylim(1.02, 0.98)
            #plt.plot(xnew, f_cubic(xnew), label='quadratic')
            #plt.legend(loc='best')
            
            #plt.draw()
            #plt.pause(0.0)
            #plt.clf()
            pass
     
          #plt.show()
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

  
df["distances"] = df["angle_corr_tail"].apply(distances)

#df["max_row_distance"] = np.NaN

summed_distances = df["distances"].mean()/4

def coord_calc(dataf):
  tail_points = dataf["angle_corr_tail"]
  #frame = dataf["index"]
  #print(frame)
  for idx, _ in enumerate(tail_points):
    if idx > 0:
      distance = math.hypot(tail_points[idx][0] - tail_points[idx-1][0], tail_points[idx][1] - tail_points[idx-1][1])
      if distance > summed_distances:
        while distance > summed_distances:
          slope, intercept, r, p, std_err = stats.linregress([tail_points[idx][0], tail_points[idx-1][0]], [tail_points[idx][1], tail_points[idx-1][1]])
          for i in range(idx, 5):
            tail_points[i][0] = tail_points[i][0]-0.0001
          tail_points[idx][1] = slope*tail_points[idx][0] + intercept           
          distance = math.hypot(tail_points[idx][0] - tail_points[idx-1][0], tail_points[idx][1] - tail_points[idx-1][1])
      else:
        while distance < summed_distances:
          slope, intercept, r, p, std_err = stats.linregress([tail_points[idx][0], tail_points[idx-1][0]], [tail_points[idx][1], tail_points[idx-1][1]])
          for i in range(idx, 5):
            tail_points[i][0] = tail_points[i][0]+0.0001
          #tail_points[idx][0] = tail_points[idx][0]+0.00001
          tail_points[idx][1] = slope*tail_points[idx][0] + intercept
          distance = math.hypot(tail_points[idx][0] - tail_points[idx-1][0], tail_points[idx][1] - tail_points[idx-1][1])

  

#df = df.reset_index()



df.apply(coord_calc, axis=1) 



print(df)
pass



# Read until video is completed
for idx_frame in range(0,final_frame,1):  
  if idx_frame in frames_numbers:  
    print(idx_frame)   
    cap.set(1, idx_frame)    
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:      
          if previous_id != frames_numbers[idx_frame][2]:
            color = next(color_cycle)
          cv2.rectangle(frame, (frames_numbers[idx_frame][0]-30,frames_numbers[idx_frame][1]-30), (frames_numbers[idx_frame][0]+30, frames_numbers[idx_frame][1]+30), color, 2)
          font = cv2.FONT_HERSHEY_SIMPLEX
          cv2.putText(frame, str(frames_numbers[idx_frame][2]),(frames_numbers[idx_frame][0]-30,frames_numbers[idx_frame][1]-30), font, 0.5, (0,0,0), 1) 
          previous_id = frames_numbers[idx_frame][2]          
          for coords in frames_numbers[idx_frame][-1]:         
            cv2.circle(frame, (coords[1], coords[0]), 2, (0, 0, 255), -1)           
          cv2.imshow('Frame', frame)
          
          tail_coordinates = df.loc[idx_frame]['angle_corr_tail']
          x_list = list(zip(*tail_coordinates))[0]
          y_list = list(zip(*tail_coordinates))[1]
          
          
          
          xnew = np.linspace(x_list[0], x_list[-1])           
            
          #f_cubic = interp1d(x_list, y_list, kind='quadratic')
          #plt.plot(xnew, f_cubic(xnew), label='quadratic')
          plt.figure(1)
          plt.ylim(1.02, 0.98)
          plt.xlim(1, 1.08)
          plt.plot(x_list, y_list, 'o', label='data')          
          model4 = np.poly1d(np.polyfit(x_list, y_list, 3))
          plt.plot(xnew, model4(xnew))
          plt.show()
          plt.pause(0.3)
                
          
          plt.figure(2)
          plt.ylim(1.02, 0.98)
          plt.xlim(1, 1.08)
          plt.plot(x_list, y_list, 'o', label='data')          
          model4 = np.poly1d(np.polyfit(x_list, y_list, 3))
          plt.plot(xnew, model4(xnew))
          plt.show()
          #plt.pause(1)  
          cv2.waitKey(3)       
          
          
          #plt.pause(2)
          #plt.clf()
                    
          x_0 = xnew[0]
          x_1 = xnew[int(len(xnew)/4)]
          x_2 = xnew[int(len(xnew)/4*2)]
          x_3 = xnew[int(len(xnew)/4*3)] 
          x_4 = xnew[-1]
                    
          slope1 = (model4(x_1) - model4(x_0))/(x_1 - x_0)
          slope2 = (model4(x_3) - model4(x_2))/(x_3 - x_2)         
          angle =  math.degrees(math.atan((slope2-slope1)/(1+(slope2*slope1))))         

          print("the angle")
          print(angle)
          print("the end angle")
            
          plt.close(1)
          
          
          
          
          
          
          
          
                     
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
df_to_analyze = df[["sequence", "angle_corr_tail"]]
df_to_analyze["head"] = df.apply(lambda x: (x.fish_head_x, x.fish_head_y), axis=1)

#df_to_analyze.rename(columns={"angle_corr_tail":"tail_coords"}, inplace=True)
df_to_analyze.rename({'angle_corr_tail': 'tail_coords'}, axis=1, inplace=True)

#df.loc[:, 'points'] = df.points.apply(lambda x: x*2)
df_to_analyze.loc[:, 'tail_coords'] = df_to_analyze.tail_coords.apply(lambda x: tuple(tuple([int((n*10000)-10000) for n in sub]) for sub in x))
df_to_analyze.to_csv('C:/Users/marcio/Documents/fish_analyzer_final' + '.csv', mode='w', index=True, header=True)








