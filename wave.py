

# Check if camera opened successfully
import cv2
import seaborn as sns
import matplotlib.pyplot as plt 

import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

with open('C:/Users/marcio/Documents/results_Ian/dict.txt', 'rb') as handle:
  frames_numbers = pickle.loads(handle.read())




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
      
          print("xxxxxxxxxx")
          print(frames_numbers[idx_frame])
      
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

            print("the angles")
            
            
            import math

            def rotate(points, radians, origin):
          
                x, y = points
                ox, oy = origin

                qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
                qy = oy + -math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)
                
                
                
               

                return qx, qy
            print(angle)
            if angle > 180:
              angle = 360-angle+180
            else:
              angle = 180-angle
          
            
            print(angle) 
              
            corrected_angle = [rotate(i, math.radians(angle), tuple_y_inv[0]) for i in tuple_y_inv]
            
       
     
            '''
            #check if x is not incremental, as fish in vertical position, then swap x and Y if true (avoiding error)            
            incremental = True
            i = 0
            max_value = 0
            while i < len(x_list):                
              if max(max_value, x_list[i]) <= max_value:
                incremental = False
                break
              else:               
                max_value = max(max_value, x_list[i])
                i += 1
            
            decremental = True
            i = 0
            min_value = 1000
            while i < len(x_list):                
              if min(min_value, x_list[i]) >= min_value:
                
                decremental = False
                break
              else:
                min_value = min(min_value, x_list[i])              
                i += 1
                
            print(x_list)
            print(y_list)           
            
            if incremental == False and decremental == False:
              x_list, y_list = y_list, x_list
              
              
              x_final = [float(i)/min(x_list) for i in x_list]             
              y_final = [float(i)/min(y_list) for i in y_list]
              
              
             
                
              print("swapped")
              print(x_list)
              print(y_list)
              print(x_final)
              print(y_final)
              print("end")

'''
    
            #x_norm = [float(i)/sum(x_final) for i in x_final]
            #y_norm = [float(i)/sum(y_final) for i in y_final]
          
           
            #x_norm = x_norm[1:]
            #y_norm = y_norm[1:]
        
            #model4 = np.poly1d(np.polyfit(x_list, x_list, 3))
            #print(model4.coefficients)
              
      
            #xnew = np.linspace(x_list[0], x_list[-1])
            
            

            # Define interpolators.
            #f_linear = interp1d(x_norm, y_norm)
            #f_cubic = interp1d(x_norm, y_norm, kind='quadratic')
            
            #plt.plot(xnew, model4(xnew), '--', color='red')

            # Plot.
            #plt.xlim([1, 1.08])
            #plt.ylim([0.1, 0.2])

            x_list_2 = list(zip(*corrected_angle))[0]
            y_list_2 = list(zip(*corrected_angle))[1]
            
            x_norm = x_list_2 #[i-x_list_2[0] for i in x_list_2]
            y_norm = y_list_2 #[i-y_list_2[0] for i in y_list_2]
            
            x_norm_2 = [float(i)/sum(x_norm) for i in x_norm]
            y_norm_2 = [float(i)/y_norm[0] for i in y_norm]
            
            xnew = np.linspace(x_norm_2[0], x_norm_2[-1])
            
            model4 = np.poly1d(np.polyfit(x_norm_2, y_norm_2, 3))
            plt.plot(xnew, model4(xnew))
            
            # Define interpolators.
            #f_linear = interp1d(x_norm, y_norm)
            #f_cubic = interp1d(x_norm, y_norm, kind='quadratic')
          
            #plt.plot(x_list, y_list, 'o', label='data')
            #plt.draw()
            #plt.plot(x_norm_2, y_norm_2, 'o', label='data')
            plt.draw()
            #plt.plot(xnew, f_linear(xnew), '-', label='linear')
            plt.xlim(0.1925, 0.21)
           # plt.ylim(0.9,1.1)
            #plt.plot(xnew, f_cubic(xnew), label='cubic')
            #plt.legend(loc='best')
            
            #plt.draw()
            plt.pause(0.5)
            #plt.clf()
            pass
     
          #plt.show()
          else:
            pass
                    
          
          

          
          
          
         
         
         
         
         

          cv2.waitKey(100)
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
cv2.waitKey(0)  
print("before ending")  
cv2.destroyAllWindows()    
