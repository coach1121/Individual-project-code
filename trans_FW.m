clc
clear all
% Set input and output folder paths
input_folder = 'C:/Users/dell/Desktop/input_folder';
output_folder = 'C:/Users/dell/Desktop/output_folder';

% Ensure the output folder exists
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Set the target frequency range (1-100Hz)
target_low_freq = 1;
target_high_freq = 100;

% Get all PNG files in the input folder
file_list = dir(fullfile(input_folder, '*.png'));

% Process each file
for i = 1:length(file_list)
    input_path = fullfile(input_folder, file_list(i).name);
    
    % Read the PNG spectrogram
    spectrogram = imread(input_path);
    
    % Get the height of the spectrogram (representing the frequency range)
    spec_height = size(spectrogram, 1);
    
    % Calculate the pixel positions of the target frequency range in the spectrogram
    target_low_pixel = round((target_low_freq / spec_height) * spec_height);
    target_high_pixel = round((target_high_freq / spec_height) * spec_height);
    
    % Crop the spectrogram to retain the target frequency range
    resampled_spectrogram = spectrogram(target_low_pixel:target_high_pixel, :,:);
    resampled_spectrogram_greyscale=rgb2gray(resampled_spectrogram);
    % Save the processed spectrogram as a PNG file to the output folder
    [~, filename, ~] = fileparts(file_list(i).name);
    output_path = fullfile(output_folder, [filename, '_resampled.png']);
    imwrite(resampled_spectrogram_greyscale, output_path);
end
