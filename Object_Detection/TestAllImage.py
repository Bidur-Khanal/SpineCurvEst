import os
from luminoth import read_image, vis_objects
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from luminoth import Detector

'''
   This is a script that is used to infer/test the object detection model.
   We use the saved checkpoints to test the the model with given test data.
   We perfrom some post-processing to reject the outliers
   Marks the vertebra location in all the test images and saves them 

'''


#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def save_image(image_directory, detection, output_path):
    all_image_boxes= []
    for filename in os.listdir(image_directory):
        if filename.endswith('jpg'):
            print (filename)
            image= read_image(image_directory+filename)
            pred_image,bounding_box = predict_vertebra(image, detection)
            all_image_boxes.append(bounding_box)
            cv2.imwrite (output_path+filename, pred_image)

    all_image_boxes= np.array(all_image_boxes) 
    np.save('boxes.npy', all_image_boxes)
            
    
def zscore(bboxes_width_height):
    mean= np.mean(bboxes_width_height,axis= 0)
    std =np. std(bboxes_width_height,axis=0)
    z_score= (bboxes_width_height-mean)/std
    return z_score

def outlier_rejection(bounding_box):
    #print (bounding_box)
    bounding_box.sort(key=lambda x: x[1])
    actual_vertebra_list=[]
    
    # remove the objects as outliers whose x-center coordinate lies outside the side edges of adjacent-connected objects/vertebra.
    # this is used to  reject the objects that are some horizontal distance apart from the connected structure of vertebra.
    for i, ver in enumerate(bounding_box):
        ver_center_x= (ver[0]+ver[2])/2
        if i == 0:
            
            below_ver1_center_x= (bounding_box[i+1][0]+bounding_box[i+1][2])/2
            below_ver2_center_x= (bounding_box[i+2][0]+bounding_box[i+2][2])/2
            if (abs(ver_center_x-below_ver1_center_x)>(ver[2]-ver[0])/2) and (abs(ver_center_x-below_ver2_center_x)>(ver[2]-ver[0])/2):
                pass
            else:
                actual_vertebra_list.append(ver)
                print (i,ver)
        elif i == (len(bounding_box)-1):
            
            above_ver1_center_x= (bounding_box[i-1][0]+bounding_box[i-1][2])/2
            above_ver2_center_x= (bounding_box[i-2][0]+bounding_box[i-2][2])/2
            if (abs(ver_center_x-above_ver1_center_x)>(ver[2]-ver[0])/2) and (abs(ver_center_x-above_ver2_center_x)>(ver[2]-ver[0])/2):
                pass
            else:
                actual_vertebra_list.append(ver)
                print (i,ver)

        else:
           
            above_ver_center_x= (bounding_box[i-1][0]+bounding_box[i-1][2])/2
            below_ver_center_x= (bounding_box[i+1][0]+bounding_box[i+1][2])/2
            if (abs(ver_center_x-above_ver_center_x)>(ver[2]-ver[0])/2) and (abs(ver_center_x-below_ver_center_x)>(ver[2]-ver[0])/2):
                pass
            else:
                actual_vertebra_list.append(ver)              
                print (i,ver)
       
    bounding_box =np.array(actual_vertebra_list)    
    width_height = bounding_box[:,2:4]-bounding_box[:,0:2]
    z_score=zscore(width_height)

    # remove the boxes with width_height zscore lower or higher than set thresholds
    # these thresholds are set after observing the zscore distribution in train data
    indices = (z_score[:,0] >-3)&(z_score[:,0]<3)&(z_score[:,1] >-3)&(z_score[:,1]<3)
    bounding_box=bounding_box[indices,:].tolist()
   
    return bounding_box

def predict_vertebra(image, detection):
    predicted_objects = detection.predict(image)

    # use only the box coordinates
    bounding_box= [b['bbox'] for b in predicted_objects]
    bounding_box= outlier_rejection(bounding_box)


    for info, box in zip(predicted_objects,bounding_box):
        vertebra=  box
        label = str(info['label'])
        prob = str(info['prob'])
        display_text = "Label:"+ label + ",Prob:"+prob
        print (display_text)
        cv2.rectangle(image,(vertebra[0],vertebra[1]),(vertebra[2],vertebra[3]),(0,255,0),5)
        print (vertebra[0],vertebra[1],vertebra[2],vertebra[3] )
  
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(image, display_text, (vertebra[0],vertebra[1]), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
    #cv2.imwrite('01-July-2019-31.jpg',image)
    return image
    

if __name__ == "__main__":
    checkpoint = "d40a34821081"
    detection = Detector(checkpoint)
    image_path = "data/cropped test/"
    output_path = "pred crop1/"
    #image = read_image(image_path + '01-July-2019-50.jpg')
    #pred_image= predict_vertebra (image,detection)
    save_image(image_path, detection, output_path)
    
  
     
 
