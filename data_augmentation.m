clc
clear all

% Input folder path (containing PNG spectrogram images)
% input_directory = 'D:\Learning Resources\Semester2\MSC Research Project\Code\pythonProject1\TEST_3classes\HW';
input_directory = 'C:/Users/dell/Desktop/LP_n_resize/';

% Output folder path
% output_directory = 'D:\Learning Resources\Semester2\MSC Research Project\Code\pythonProject1\TEST_3classes\HWa';

% Create the output folder (if it doesn't exist)
% if ~exist(output_directory, 'dir')
%     mkdir(output_directory);
% end

% List all PNG image files in the input folder
image_files = dir(fullfile(input_directory, '*.png'));

augmentation_factor = 2;

for i = 1:length(image_files)
    % Read the PNG image
    image_path = fullfile(input_directory, image_files(i).name);
    image = imread(image_path);
    
    %% Apply frequency domain effects 
    augmented_image = fliplr(image);
%     for j = 1:augmentation_factor
%         % Construct the output image file's save path
%         output_image_path = fullfile(output_directory, ...
%             sprintf('%s_aug%d.png', image_files(i).name, j));
%         
%         % Save the augmented image
%         imwrite(augmented_image, output_image_path);
%     end
    for j = 1:augmentation_factor
        % Construct the output image file's save path
        % Save the augmented image
        filepath = pwd; % Save the current working directory
        cd('C:\Users\dell\Desktop\LP_n_resize_ag\1') % Switch the current working directory to the specified folder
        imwrite(augmented_image, fullfile(sprintf('%s_aug%d.png', image_files(i).name, j + augmentation_factor)))
        cd(filepath) % Switch back to the original working directory
    end
    
    %% Apply time stretching 
    augmented_image = flipud(image);
    for j = 1:augmentation_factor
        % Construct the output image file's save path
        % Save the augmented image
        filepath = pwd; % Save the current working directory
        cd('C:\Users\dell\Desktop\LP_n_resize_ag\1') % Switch the current working directory to the specified folder
        imwrite(augmented_image, fullfile(sprintf('%s_aug%d.png', image_files(i).name, j + augmentation_factor)))
        cd(filepath) % Switch back to the original working directory
    end
    
    %% Apply amplitude scaling 
    augmented_image = imadjust(image, [0.2, 0.8], []);
%     for j = 1:augmentation_factor
%         % Construct the output image file's save path
%         output_image_path = fullfile(output_directory, ...
%             sprintf('%s_aug%d.png', image_files(i).name, j + 2 * augmentation_factor));
%         
%         % Save the augmented image
%         imwrite(augmented_image, output_image_path);
%     end
    for j = 1:augmentation_factor
        % Construct the output image file's save path
        % Save the augmented image
        filepath = pwd; % Save the current working directory
        cd('C:\Users\dell\Desktop\LP_n_resize_ag\1') % Switch the current working directory to the specified folder
        imwrite(augmented_image, fullfile(sprintf('%s_aug%d.png', image_files(i).name, j + augmentation_factor)))
        cd(filepath) % Switch back to the original working directory
    end
end

%% Test
filepath = pwd; % Save the current working directory
cd('C:\Users\dell\Desktop\LP_n_resize_ag\1') % Switch the current working directory to the specified folder
imwrite(image, '2.png')
cd(filepath) % Switch back to the original working directory
