clc
clear all

input_folder = 'C:/Users/dell/Desktop/LP/19';
output_folder = 'C:/Users/dell/Desktop/LP_n_resize/19';

% Collect the details of the PNG files in the input folder
dirdata = dir(fullfile(input_folder, '*.png'));

% Desired size for resizing images
desiredSize = [224, 224];

% Loop over all of the .png files in the input folder
for k = 1:numel(dirdata)
    disp(k)                 % Display count

    % Get name of the next file
    png_file = dirdata(k).name;

    % Create the full file name for the input PNG file
    input_file = fullfile(input_folder, png_file);

    % Read the PNG image
    original_image = imread(input_file);

    % Resize the image to the desired size using bilinear interpolation
    resized_image = imresize(original_image, desiredSize, 'bilinear');

    % Nomoralization
    resized_image = double(resized_image) / 255;


    % Create the full file name for the output PNG file
    [~, name, ext] = fileparts(png_file);
    output_file = fullfile(output_folder, [name, '_resized', ext]);

    % Save the resized image in the output folder
    imwrite(resized_image, output_file, 'png');
end
