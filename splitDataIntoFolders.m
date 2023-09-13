function splitDataIntoFolders(sourceFolder, destFolder)
% This function splits files from a source folder into Train, Val, and Test folders
% at the destination path, following an 8:1:1 ratio.

% Check if the source folder exists
if ~exist(sourceFolder, 'dir')
    error('The source folder does not exist!');
end

% Create Train, Val, and Test folders at the destination
trainFolder = fullfile(destFolder, 'Train');
valFolder = fullfile(destFolder, 'Val');
testFolder = fullfile(destFolder, 'Test');

if ~exist(trainFolder, 'dir')
    mkdir(trainFolder);
end

if ~exist(valFolder, 'dir')
    mkdir(valFolder);
end

if ~exist(testFolder, 'dir')
    mkdir(testFolder);
end

% Get all files from the source folder
fileList = dir(fullfile(sourceFolder, '*'));
% Filter out directories and retain only files
fileList = fileList(~[fileList.isdir]);

% Shuffle the file list randomly
randIndices = randperm(length(fileList));

% Allocate files according to the 8:1:1 ratio
numFiles = length(fileList);
numTrain = round(0.8 * numFiles);
numVal = round(0.1 * numFiles);
% Remaining files go to Test

% Copy files to their respective folders
for i = 1:numFiles
    if i <= numTrain
        copyfile(fullfile(sourceFolder, fileList(randIndices(i)).name), trainFolder);
    elseif i <= numTrain + numVal
        copyfile(fullfile(sourceFolder, fileList(randIndices(i)).name), valFolder);
    else
        copyfile(fullfile(sourceFolder, fileList(randIndices(i)).name), testFolder);
    end
end

disp('File distribution complete!');

end
