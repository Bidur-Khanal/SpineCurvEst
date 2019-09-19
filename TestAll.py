
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
from luminoth import read_image, vis_objects
from luminoth import Detector
from Landmark_Detection.model import DenseNet

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

'''
   This is a script that is used to infer/test the two stage approach, vertebra detection followed by landmark detection
   We use the saved checkpoints to test the the model with given test data.
   We perfrom some post-processing to reject the outliers
   find the landmarks in the image and save

'''


def generate_landmark_csv(image_directory, filename_csv, detection, model, csv_save_path, img_save_path):
    filename_labels = pd.read_csv(filename_csv, header=None)
    df = pd.DataFrame()
    for i, names in enumerate(filename_labels.iloc[:, 0]):
        image = read_image(image_directory + names)
        landmarks = predict_vertebra(image, detection, model)
        df = df.append([landmarks])
        save_landmarks_image(image,landmarks,img_save_path +names)

    df.to_csv(csv_save_path, index=False, header=False)


def predict_vertebra(image, detection, model):
    predicted_objects = detection.predict(image)
    landmarks = []
    # use only the box coordinates
    bounding_box= [b['bbox'] for b in predicted_objects]

    #reject bounding boxes that are outliers 
    bounding_box=outlier_rejection(bounding_box)

    # sort the vertebra in order, T1, T2, .... L5
    bounding_box.sort(key=lambda x: x[1])
    
    if len(bounding_box)<17:
        missing_vertebra = 17-len(bounding_box)
        print ("vertebra less than 17,i.e", len(bounding_box))
        
        bounding_box= bounding_box+bounding_box[-missing_vertebra:]
        print (bounding_box)
    if len(bounding_box)>17:
        bounding_box= bounding_box[:17]

    for vertebra in bounding_box:
        img = np.copy(image)
        # crop the patch from the original image
        patch = img[vertebra[1]:vertebra[3], vertebra[0]:vertebra[2]]
        # predict local landmark within the patch
        local_landmark = predict_landmark(patch, model)
        # convert the landmark's local position to the global position with respect to original image
        landmark_global_pos = np.zeros_like(local_landmark)
        landmark_global_pos[0:8:2] = (local_landmark[0:8:2] + vertebra[0])/image.shape[1]
        landmark_global_pos[1:8:2] = (local_landmark[1:8:2] + vertebra[1])/image.shape[0]
        landmarks.append(landmark_global_pos)
    #change the order of coordinates within list; all x coordinates first followed by y coordinates 
    landmarks = np.array(landmarks).ravel().tolist()
    landmarks = landmarks[::2] + landmarks[1::2]
    #print (landmarks)
    return landmarks


def predict_landmark(image, model):

    im = cv2.resize(image,(200,120),interpolation=cv2.INTER_AREA)
    im = np.delete(im, [1, 2], axis=2)
    im = np.array(im) / 255.0
    im = np.expand_dims(im, axis=0)
    print(im.shape)
    lmarks = model.predict(im)
    lmarks = lmarks[0]
    print(lmarks)
    lmarks[0:8:2] = lmarks[0:8:2] * image.shape[1]
    lmarks[1:8:2] = lmarks[1:8:2] * image.shape[0]
    return np.around(lmarks)


def load_densenet_model(model_path):
    model = DenseNet(dense_blocks=5, dense_layers=-1, growth_rate=8, dropout_rate=0.2, bottleneck=True, compression=1.0, weight_decay=1e-4, depth=40)
    model.load_weights(model_path)
    return model


def save_landmarks_image(img, landmark, img_output_path):

    for m in range(0, int(len(landmark ) /2)):
        cv2.circle(img, (int(img.shape[1]*landmark[m]), int(img.shape[0]*landmark[m + int(len(landmark ) /2)])), 10, (0, 0, 255), -1)

    cv2.imwrite(img_output_path ,img)

def zscore(bboxes_width_height):
    mean= np.mean(bboxes_width_height,axis= 0)
    std =np. std(bboxes_width_height,axis=0)
    z_score= (bboxes_width_height-mean)/std
    return z_score

def outlier_rejection(bounding_box):

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
               
        elif i == (len(bounding_box)-1):
            
            above_ver1_center_x= (bounding_box[i-1][0]+bounding_box[i-1][2])/2
            above_ver2_center_x= (bounding_box[i-2][0]+bounding_box[i-2][2])/2
            
            if (abs(ver_center_x-above_ver1_center_x)>(ver[2]-ver[0])/2) and (abs(ver_center_x-above_ver2_center_x)>(ver[2]-ver[0])/2):
                pass
            else:
                actual_vertebra_list.append(ver)
                

        else:
           
            above_ver_center_x= (bounding_box[i-1][0]+bounding_box[i-1][2])/2
            below_ver_center_x= (bounding_box[i+1][0]+bounding_box[i+1][2])/2
            
            if (abs(ver_center_x-above_ver_center_x)>(ver[2]-ver[0])/2) and (abs(ver_center_x-below_ver_center_x)>(ver[2]-ver[0])/2):
                pass
            else:
                actual_vertebra_list.append(ver)              
                
       
    bounding_box =np.array(actual_vertebra_list)    
    width_height = bounding_box[:,2:4]-bounding_box[:,0:2]
    z_score=zscore(width_height)

    # remove the boxes with width_height zscore lower or higher than set thresholds
    # these thresholds are set after observing the zscore distribution in train data
    indices = (z_score[:,0] >-3)&(z_score[:,0]<3)&(z_score[:,1] >-3)&(z_score[:,1]<3)
    bounding_box=bounding_box[indices,:].tolist()
   
    return bounding_box

    
    
    
   
    




if __name__ == "__main__":
    checkpoint = "d40a34821081"
    detection = Detector(checkpoint)
    model = load_densenet_model("Landmark_Detection/outputs/model-518.h5")
    image_path = "data/cropped test/"
    filename_csv = "data/labels/test_filenames.csv"
    '''image = read_image(image_path + '01-July-2019-7.jpg')
    landmarks = predict_vertebra(image, detection, model)
    img_output_path= 'results/detected_landmark.jpg'
    save_landmarks_image(image,landmarks, img_output_path)'''
    generate_landmark_csv(image_path ,filename_csv ,detection ,model ,'test_results/test_landmarks.csv' ,'test_results/')
