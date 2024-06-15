% Define paths
imageDir = 'data_for_moodle/images_256'; % Path where the original images are stored
labelDir = 'data_for_moodle/labels_256'; % Path where the corresponding labels are stored
unlabeledImageDir = 'data_for_moodle/unlabelled'; % Path to store images without labels

% Create the directory for unlabeled images if it doesn't exist
if ~exist(unlabeledImageDir, 'dir')
    mkdir(unlabeledImageDir);
end

% Read filenames
imageFiles = dir(fullfile(imageDir, '*.jpg')); % Adjust the extension as needed
labelFiles = dir(fullfile(labelDir, '*.png')); % Adjust the extension as needed

%%
% Extract the base names without extension
imageNames = {imageFiles.name};
imageBaseNames = cellfun(@(x) erase(x, '.jpg'), imageNames, 'UniformOutput', false); % Change '.jpg' as needed

labelNames = {labelFiles.name};
labelBaseNames = cellfun(@(x) erase(x, '.png'), labelNames, 'UniformOutput', false); % Change '.png' as needed

%%
% Find images without matching labels
hasLabel = ismember(imageBaseNames, labelBaseNames);
unlabeledImages = imageNames(~hasLabel);

%%
% Move files to the 'unlabeledImageDir'
for k = 1:length(unlabeledImages)
    movefile(fullfile(imageDir, unlabeledImages{k}), fullfile(unlabeledImageDir, unlabeledImages{k}));
end

%% Print the number of images in each folder
numImagesInImageDir = numel(dir(fullfile(imageDir, '*.jpg'))); % Adjust the extension as needed
numImagesInLabelDir = numel(dir(fullfile(labelDir, '*.png'))); % Adjust the extension as needed
numImagesInUnlabeledDir = numel(dir(fullfile(unlabeledImageDir, '*.jpg'))); % Adjust the extension as needed

fprintf('%d images were moved to the folder: %s\n', length(unlabeledImages), unlabeledImageDir);
fprintf('%d images remain in the original images folder: %s\n', numImagesInImageDir, imageDir);
fprintf('%d labels are in the labels folder: %s\n', numImagesInLabelDir, labelDir);
fprintf('%d images are in the unlabeled images folder: %s\n', numImagesInUnlabeledDir, unlabeledImageDir);