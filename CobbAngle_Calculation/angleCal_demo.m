
ap_num = 68;

N =98 %number of test images

% get image
folder_im = 'C:\Users\Brinda Khanal\Documents\Bidur Git Repo\Spine_Challenge\test cropped\crop\';
files_im = [folder_im  '*.jpg'];

dirOutput_im = dir(files_im);
fileNames_im = {dirOutput_im.name}';


CobbAn_ap = [];
CobbAn_lat = [];
landmarks_ap = [];
landmarks_lat = [];
landmarks = csvread('C:\Users\Brinda Khanal\Documents\Bidur Git Repo\Spine_Challenge\Cobb angle Calculation\test_landmarks.csv');



for k=1:N
    %get images
    l = [folder_im fileNames_im{k}];
    im = imread(l);
    [H,W] = size(im);
    
    %get landmarks
    
    p2 = [landmarks(k,1:68) *W ; landmarks(k,69:136) * H]';
    
    % use the code below for variable landmarks, i.e if its greater or less than 136
    %landmarks_nonzero=nonzeros(landmarks(k,:))
    %ap_num=size(landmarks_nonzero)(1)/2
    %p2=[landmarks_nonzero(1:ap_num)*W;landmarks_nonzero(ap_num+1:2*ap_num)*H]';
    %p2=reshape(p2,ap_num,2)
    
    
    vnum = ap_num / 4;
    
 
    cob_angles = zeros(1,3);
        
    figure,imshow(im)
    title('Image');
    hold on
    
 
    mid_p_v = zeros(size(p2,1)/2,2);
    for n=1:size(p2,1)/2
        mid_p_v(n,:) = (p2(n*2,:) + p2((n-1)*2+1,:))/2;
          % disp ('midppoint')
    end
    
    
    %calculate the middle vectors & plot the labeling lines
    mid_p = zeros(size(p2,1)/2,2);
    for n=1:size(p2,1)/4
        mid_p((n-1)*2+1,:) = (p2(n*4-1,:) + p2((n-1)*4+1,:))/2;
        mid_p(n*2,:) = (p2(n*4,:) + p2((n-1)*4+2,:))/2;
    end
    
    
    
    %plot the midpoints
    plot(mid_p(:,1),mid_p(:,2),'y.','MarkerSize',20);
    pause(1)
    
    
    vec_m = zeros(size(mid_p,1)/2,2);
    for n=1:size(mid_p,1)/2
        vec_m(n,:) = mid_p(n*2,:) - mid_p((n-1)*2+1,:);
        %plot the midlines
        plot([mid_p(n*2,1),mid_p((n-1)*2+1,1)],...
            [mid_p(n*2,2),mid_p((n-1)*2+1,2)],'Color','r','LineWidth',2);
    end
    
    mod_v = power(sum(vec_m .* vec_m, 2),0.5);
    dot_v = vec_m * vec_m';
    
    %calculate the Cobb angle
    angles = acos(dot_v./(mod_v * mod_v'));
    [maxt, pos1] = max(angles);
    [pt, pos2] = max(maxt);
    pt = real(pt/pi*180);
    cob_angles(1) = pt;
    
    %plot the selected lines

    plot([mid_p(pos2*2,1),mid_p((pos2-1)*2+1,1)],...
        [mid_p(pos2*2,2),mid_p((pos2-1)*2+1,2)],'Color','g','LineWidth',2);
    plot([mid_p(pos1(pos2)*2,1),mid_p((pos1(pos2)-1)*2+1,1)],...
        [mid_p(pos1(pos2)*2,2),mid_p((pos1(pos2)-1)*2+1,2)],'Color','g','LineWidth',2);
    
    if ~isS(mid_p_v) % 'S'
        
        disp ('The Vertebra is S shaped')
        
        mod_v1 = power(sum(vec_m(1,:) .* vec_m(1,:), 2),0.5);
        mod_vs1 = power(sum(vec_m(pos2,:) .* vec_m(pos2,:), 2),0.5);
        mod_v2 = power(sum(vec_m(vnum,:) .* vec_m(vnum,:), 2),0.5);
        mod_vs2 = power(sum(vec_m(pos1(pos2),:) .* vec_m(pos1(pos2),:), 2),0.5);
        
        dot_v1 = vec_m(1,:) * vec_m(pos2,:)';
        dot_v2 = vec_m(vnum,:) * vec_m(pos1(pos2),:)';
        
        mt = acos(dot_v1./(mod_v1 * mod_vs1'));
        tl = acos(dot_v2./(mod_v2 * mod_vs2'));
        
        mt = real(mt/pi*180);
        cob_angles(2) = mt;
        tl = real(tl/pi*180);
        cob_angles(3) = tl;
        
    else
        
    % max angle in the upper part
        if (mid_p_v(pos2*2,2) + mid_p_v(pos1(pos2)*2,2)) < size(im,1)
            
            %calculate the Cobb angle (upside)
            mod_v_p = power(sum(vec_m(pos2,:) .* vec_m(pos2,:), 2),0.5);
            mod_v1 = power(sum(vec_m(1:pos2,:) .* vec_m(1:pos2,:), 2),0.5);
            dot_v1 = vec_m(pos2,:) * vec_m(1:pos2,:)';
            
            
            angles1 = acos(dot_v1./(mod_v_p * mod_v1'));
            [CobbAn1, pos1_1] = max(angles1);
            mt = real(CobbAn1/pi*180);
            cob_angles(2) = mt;
            
            plot([mid_p(pos1_1*2,1),mid_p((pos1_1-1)*2+1,1)],...
                [mid_p(pos1_1*2,2),mid_p((pos1_1-1)*2+1,2)],'Color','g','LineWidth',2);
            
            
            %calculate the Cobb angle?downside?
            mod_v_p2 = power(sum(vec_m(pos1(pos2),:) .* vec_m(pos1(pos2),:), 2),0.5);
            mod_v2 = power(sum(vec_m(pos1(pos2):vnum,:) .* vec_m(pos1(pos2):vnum,:), 2),0.5);
            dot_v2 = vec_m(pos1(pos2),:) * vec_m(pos1(pos2):vnum,:)';
            
            angles2 = acos(dot_v2./(mod_v_p2 * mod_v2'));
            [CobbAn2, pos1_2] = max(angles2);
            tl = real(CobbAn2/pi*180);
            cob_angles(3) = tl;
            
            pos1_2 = pos1_2 + pos1(pos2) - 1;
            plot([mid_p(pos1_2*2,1),mid_p((pos1_2-1)*2+1,1)],...
                [mid_p(pos1_2*2,2),mid_p((pos1_2-1)*2+1,2)],'Color','g','LineWidth',2);
            
        else
            %calculate the Cobb angle (upside)
            mod_v_p = power(sum(vec_m(pos2,:) .* vec_m(pos2,:), 2),0.5);
            mod_v1 = power(sum(vec_m(1:pos2,:) .* vec_m(1:pos2,:), 2),0.5);
            dot_v1 = vec_m(pos2,:) * vec_m(1:pos2,:)';
            
            
            angles1 = acos(dot_v1./(mod_v_p * mod_v1'));
            [CobbAn1, pos1_1] = max(angles1);
            mt = real(CobbAn1/pi*180);
            cob_angles(2) = mt;
            
            plot([mid_p(pos1_1*2,1),mid_p((pos1_1-1)*2+1,1)],...
                [mid_p(pos1_1*2,2),mid_p((pos1_1-1)*2+1,2)],'Color','g','LineWidth',2);
            
            
            %calculate the Cobb angle (upper upside)
            mod_v_p2 = power(sum(vec_m(pos1_1,:) .* vec_m(pos1_1,:), 2),0.5);
            mod_v2 = power(sum(vec_m(1:pos1_1,:) .* vec_m(1:pos1_1,:), 2),0.5);
            dot_v2 = vec_m(pos1_1,:) * vec_m(1:pos1_1,:)';
            
            angles2 = acos(dot_v2./(mod_v_p2 * mod_v2'));
            [CobbAn2, pos1_2] = max(angles2);
            tl = real(CobbAn2/pi*180);
            cob_angles(3) = tl;
            
            %pos1_2 = pos1_2 + pos1(pos2) - 1;
            plot([mid_p(pos1_2*2,1),mid_p((pos1_2-1)*2+1,1)],...
                [mid_p(pos1_2*2,2),mid_p((pos1_2-1)*2+1,2)],'Color','g','LineWidth',2);
        end
    end
    
    
    output = [ num2str(k) ': the Cobb Angles(PT, MT, TL/L) are '  num2str(pt) ', ' num2str(mt) ' and '  num2str(tl) ...
        ', and the two most tilted vertebrae are ' num2str(pos2) ' and ' num2str(pos1(pos2)) '.\n'];
   
    fprintf(output);
   
    close all
   
    CobbAn_ap = [CobbAn_ap ; cob_angles]; %cobb angles
    

end

% write to csv file
csvwrite('test_angles_ap13.csv',CobbAn_ap);




