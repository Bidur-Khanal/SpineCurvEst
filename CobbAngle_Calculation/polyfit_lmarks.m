
%The script polyfits the landmarks. The landmarks is to be loaded from csv
%file, and total 136 coordinates including x and y is expected.

clear all
close all

img_path = 'landmark_prediction_network/cropped test/';
all_img_list= dir(strcat(img_path , '\*.jpg'));
addpath(genpath(img_path));
lmarks = csvread('results/test_landmarks.csv');
save_name = 'test_lmarks_polyfit.csv';     
save_lm_all= zeros(length(all_img_list),136);    
polyfit_order = 6;
vis_lmarks = 1;     %Flag to visualize lmarks

for i = 1:length(all_img_list)
    save_lm = zeros(1,136);      %68x2 Coordinates (x and y)
    lm_first = lmarks(i,:);
    lm_first(lm_first==0)=[];     %Remove 0 elements from csv
    length_x = length(lm_first);
    
    x = lm_first(1:length_x/2);
    y = lm_first(length_x/2+1:end);
    
    x_left= x(1:2:end);  y_left = y(1:2:end);
    x_right= x(2:2:end); y_right = y(2:2:end);
  
    disp (all_img_list(i).name);
    img_img = imread(strcat(img_path, all_img_list(i).name));
    img_img_gray=rgb2gray(img_img);
    [col,row ]=size(img_img_gray); 
    
    x_left= x_left .* row ; y_left = y_left .* col;
    x_right= x_right .* row ;  y_right = y_right .* col;  %Multiply by image size

    %Polyfit on the landmarks of desired order. 
    polyfit_test_left = polyfit(y_left,x_left,polyfit_order);
    polyfit_result_left = polyval(polyfit_test_left,y_left);
    polyfit_test_right = polyfit(y_right,x_right,polyfit_order );
    polyfit_result_right = polyval(polyfit_test_right,y_right);    
   
%     %Visualize image & Polyfit landmarks
%----------------------------------------------------------------
    if vis_lmarks
        imshow(img_img_gray)
        hold on;
        plot(polyfit_result_left,y_left,  'g*', 'LineWidth', 1, 'MarkerSize', 2);
        plot(x_left,y_left , 'r*', 'LineWidth', 1, 'MarkerSize', 2);
        plot(polyfit_result_right,y_right,  'g*', 'LineWidth', 1, 'MarkerSize', 2);
        plot(x_right,y_right , 'r*', 'LineWidth', 1, 'MarkerSize', 2);
        hold off;
        w = waitforbuttonpress;
    end
    
%----------------------------------------------------------------
%     Save the landmarks after normalizing it between 0 and 1. 
    x_left_save= polyfit_result_left ./ row ;   y_left_save = y_left ./ col;
    x_right_save= polyfit_result_right ./ row ; y_right_save = y_right ./ col;

    i_count_index = 1;
    
    for i_count = 1:length(x_left)
        save_lm(i_count_index)= x_left_save(i_count);
        save_lm(i_count_index+1)= x_right_save(i_count);
        i_count_index = i_count_index +2;
    end
     i_count_index = length(x_left)*2+ 1;
    for i_count = 1:length(x_left)
        save_lm(i_count_index)= y_left_save(i_count);
        save_lm(i_count_index+1)= y_right_save(i_count);
        i_count_index = i_count_index +2;
    end
    
    save_lm_all(i,:) = save_lm;  
end
csvwrite(save_name, save_lm_all)
