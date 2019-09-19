1.sample submission: 
From ObjectDet_LandmarkPred 
increased bounding box by (50,10) pixels, didn't resize the boxes before feeding to densetnet for corner landmark detection.
SMAPE - 39.26

2.sample submission2:
From ObjectDet_LandmarkPred
increased bounding box by (50,10) pixels, resize the boxes to training size before feeding to densenet for corner landmark detection.
SMAPE - 33.3

3.sample submission3:
From ObjectDet_LandmarkPred
Added postprocessing to (2.) to remove larger bounding boxes as outlier using Z-score.
SMAPE - 33.8, no improvements, looks like object detection itself has to be accurate

4. sample_submission4:
From ObjectDet_LandmarkPred2
(10,10) increased bounding box, resized the boxes to 200,120 before feeding to densenet.
SMAPE- 37.95
looks (50,10) is better than (10,10) i.e increasing the bounding box slightly more will result in better object detection and landmark estimation.


5.sample_submission 6:
cropped the test images (removed head and below the waist region)
SMAPE - 26.7

6. sample_submission 10:
remove z-score based outlier rejection (all detected objects are are preserved, except for side outliers)
use all the vertebras to predict the angle.
SMAPE- 27.23

7.sample_submission 11:
same as the sample submission 10, but instead of using all the detected vertebras, we have limited the vertebras to 17.
i.e if greater than 17, delete vertebrae below it, if less than 17 copy the last vertebrae to make 17
SMAPE - 26.4

8.sample_submission11-polyfit:
after the landmark detection on test_landmarks 11, 9th order polyfit function is applied to smoothen the spine curvature.
improved the SMAPE - score 26.36

9. sample_submission11-polyfit_order7:
used the polyfit order 7 on test_landmarks 11
SMAPE improved : 26.05

10. sample_submission11-polyfit_order6:
used the polyfit order 6 on test_landmarks 11
SMAPE improved : 25.69






