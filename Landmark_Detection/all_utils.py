from __future__ import print_function
import numpy as np
from PIL import Image



def get_data_generator(df, indices, for_training, image_path, batch_size=16):
    images, landmarks = [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, lmark= r['image_id'], r[['x1','y1','x2','y2','x3','y3','x4','y4']].values
            im = Image.open(image_path + file)
            im=np.array(im)
            im= np.delete(im,[1,2],axis=2)
            
            im = np.array(im) / 255.0
            images.append(im)

            landmarks.append(lmark)
            if len(images) >= batch_size:
               
                yield np.array(images), np.array(landmarks)
                images, landmarks = [], []
        if not for_training:
            break
            
            

            
            
            
