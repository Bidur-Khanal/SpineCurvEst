#import important modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
import math as mth
from PIL import Image, ImageDraw
import seaborn as sns

# compare the cobb angles

def plot_angles(csv_path_gnd, csv_path_est):
    angles_gnd = pd.read_csv(csv_path_gnd)
    angles_est = pd.read_csv(csv_path_est)

    MT_gnd = angles_gnd['MT'].values
    MT_est = angles_est['MT'].values

    PT_gnd = angles_gnd['PT'].values
    PT_est = angles_est['PT'].values

    TL_gnd = angles_gnd['TL'].values
    TL_est = angles_est['TL'].values
    print (len(MT_gnd), len(MT_est))

    MT_MSE = np.square(np.subtract(MT_gnd, MT_est)).mean()
    PT_MSE = np.square(np.subtract(PT_gnd, PT_est)).mean()
    TL_MSE = np.square(np.subtract(TL_gnd, TL_est)).mean()

    MT_MAE = np.abs(MT_gnd - MT_est).mean()
    PT_MAE = np.abs(PT_gnd - PT_est).mean()
    TL_MAE = np.abs(TL_gnd - TL_est).mean()

    print ("Mean Squared Error", MT_MSE, PT_MSE, TL_MSE)
    print ("Absolute Error", MT_MAE, PT_MAE, TL_MAE)

    # set width of bar
    barWidth = 1

    plt.figure(figsize=(200, 100))
    # Set position of bar on X axis
    r1 = np.arange(len(MT_gnd))
    r2 = [x + barWidth for x in r1]
    print (r1)
    print (r2)

    # Make the plot

    ax1= plt.subplot(311)
    ax1.bar(r1, MT_gnd-MT_est, color='blue', width=barWidth, edgecolor='white')
    # ax1.bar(r2, MT_est, color='blue', width=barWidth, edgecolor='white', label='Estimated Angles')
    # Add xticks on the middle of the group bars
    ax1.set_xlabel('Image Index')
    ax1.set_xticks([r + barWidth for r in range(len(MT_gnd))])
    ax1.set_ylabel('Error in Angle (MT)')
    ax1.text(0.5, 0.75, 'MAE = ' + str(MT_MAE) + '\n' + 'MSE = ' + str(MT_MSE),transform=ax1.transAxes)
    #plt.setp(ax1.get_xticklabels(), visible=False)

    ax2=plt.subplot(312)
    ax2.bar(r1, PT_gnd-PT_est, color='blue', width=barWidth, edgecolor='white')
    #ax2.bar(r2, PT_est, color='blue', width=barWidth, edgecolor='white', label='Estimated Angles')
    # Add xticks on the middle of the group bars
    ax2.set_xlabel('Image Index',)
    ax2.set_xticks([r + barWidth for r in range(len(PT_gnd))])
    ax2.set_ylabel('Error in Angle (PT)')
    ax2.text(0.5, 0.75, 'MAE = ' + str(PT_MAE) + '\n' + 'MSE = ' + str(PT_MSE), transform=ax2.transAxes)

    ax3= plt.subplot(313)
    ax3.bar(r1, TL_gnd-TL_est, color='blue', width=barWidth, edgecolor='white')
    #ax3.bar(r2, TL_est, color='blue', width=barWidth, edgecolor='white', label='Estimated Angles')
    # Add xticks on the middle of the group bars
    ax3.set_xlabel('Image Index')
    ax3.set_xticks([r + barWidth for r in range(len(TL_gnd))])
    ax3.set_ylabel('Error in Angle (TL)')
    ax3.text(0.5, 0.75, 'MAE = '+str(TL_MAE) +'\n' + 'MSE = '+ str(TL_MSE),transform=ax3.transAxes)

    plt.legend()
    plt.show()




csv_path_gnd= "C:/Users/Brinda Khanal/Downloads/scoliosis xray Single View/boostnet_labeldata/labels/training/angles.csv"
csv_path_est = "C:/Users/Brinda Khanal/Documents/Bidur Git Repo/Spine_Challenge/angles_ap.csv"

plot_angles (csv_path_gnd,csv_path_est)