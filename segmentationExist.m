%% Setup paths for images and labels
dataSetDir = 'data_for_moodle';
imageDir = fullfile(dataSetDir, 'images_256');
labelDir = fullfile(dataSetDir, 'labels_256');

% Define the directories for the split datasets
trainImageDir = fullfile(dataSetDir, 'training');
valImageDir = fullfile(dataSetDir, 'validation');
testImageDir = fullfile(dataSetDir, 'testing');

trainLabelDir = fullfile(dataSetDir, 'training_labels');
valLabelDir = fullfile(dataSetDir, 'validation_labels');
testLabelDir = fullfile(dataSetDir, 'test_labels');

%% Create the directories if they don't exist
folders = {trainImageDir, valImageDir, testImageDir, trainLabelDir, valLabelDir, testLabelDir};
for i = 1:numel(folders)
    if ~exist(folders{i}, 'dir')
        mkdir(folders{i});
    end
end

%% Load images and labels
imds = imageDatastore(imageDir);
classNames = ["flower", "background"];
pixelLabelID = {1, 3};  % 1 is for flower and 3 is background

%% To check if the toolbox is available
assert(exist('pixelLabelDatastore', 'file') == 2, 'Image Processing Toolbox might be missing.');

%% Create a PixelLabelDatastore for the ground truth labels
pxds = pixelLabelDatastore(labelDir, classNames, pixelLabelID);

%% Split data into training, validation, and testing sets
% Count and shuffle indices to maintain the correspondence between images and labels
totalNumImages = numel(imds.Files);
shuffledIndices = randperm(totalNumImages);

% Calculate indices for each set
numTrain = floor(0.7 * totalNumImages);
numVal = floor(0.15 * totalNumImages);
numTest = totalNumImages - numTrain - numVal;

% Move images and labels to their respective directories
for i = 1:numTrain
    copyfile(imds.Files{shuffledIndices(i)}, trainImageDir);
    copyfile(pxds.Files{shuffledIndices(i)}, trainLabelDir);
end
for i = numTrain+1:numTrain+numVal
    copyfile(imds.Files{shuffledIndices(i)}, valImageDir);
    copyfile(pxds.Files{shuffledIndices(i)}, valLabelDir);
end
for i = numTrain+numVal+1:totalNumImages
    copyfile(imds.Files{shuffledIndices(i)}, testImageDir);
    copyfile(pxds.Files{shuffledIndices(i)}, testLabelDir);
end

%% Update Datastores to Point to New Folders
imdsTrain = imageDatastore(trainImageDir);
imdsVal = imageDatastore(valImageDir);
imdsTest = imageDatastore(testImageDir);

pxdsTrain = pixelLabelDatastore(trainLabelDir, classNames, pixelLabelID);
pxdsVal = pixelLabelDatastore(valLabelDir, classNames, pixelLabelID);
pxdsTest = pixelLabelDatastore(testLabelDir, classNames, pixelLabelID);

%% Prepare the training and validation data
trainingData = pixelLabelImageDatastore(imdsTrain, pxdsTrain);
validationData = pixelLabelImageDatastore(imdsVal, pxdsVal);

%% Create the U-Net network
imageSize = [256 256 3]; % Size of the input images
numClasses = 2;
lgraph = unetLayers(imageSize, numClasses);

%% Specify training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 8, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', validationData, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'VerboseFrequency', 2);

%% Train the network
net = trainNetwork(trainingData, lgraph, options);

%% Save the trained network
save('segmentexistnet.mat', 'net');

%% Load the training model for testing
load('segmentexistnet.mat', 'net');

%% Define the colormap for the classes
classNames = ["flower", "background"];
numClasses = numel(classNames);
cmap = [1 0 0; 0 0 1];  % Red for 'flower', Blue for 'background'

%% Initialize arrays to accumulate IoU, Dice, and confusion matrix values
totalIoU = zeros(numClasses, 1);
totalDice = zeros(numClasses, 1);
totalTP = zeros(numClasses, 1);
totalFP = zeros(numClasses, 1);
totalFN = zeros(numClasses, 1);
totalTN = zeros(numClasses, 1);
numImages = numel(imdsTest.Files);

%% Process each image in the test dataset
for idx = 1:numImages
    % Read the test image and corresponding label
    I = readimage(imdsTest, idx);
    expectedResult = readimage(pxdsTest, idx);

    % Perform semantic segmentation using the trained U-Net model
    C = semanticseg(I, net);

    % Calculate the Intersection over Union (IoU) and Dice coefficient for this image
    iou = jaccard(C, expectedResult);
    totalIoU = totalIoU + iou(:);
    dice = 2 * iou ./ (1 + iou);
    totalDice = totalDice + dice(:);

    % Calculate confusion matrix components (TP, FP, FN, TN)
    for c = 1:numClasses
        classMask = expectedResult == classNames(c);
        predictedMask = C == classNames(c);

        TP = sum((predictedMask == 1) & (classMask == 1), 'all');
        FP = sum((predictedMask == 1) & (classMask == 0), 'all');
        FN = sum((predictedMask == 0) & (classMask == 1), 'all');
        TN = sum((predictedMask == 0) & (classMask == 0), 'all');

        totalTP(c) = totalTP(c) + TP;
        totalFP(c) = totalFP(c) + FP;
        totalFN(c) = totalFN(c) + FN;
        totalTN(c) = totalTN(c) + TN;
    end
end

% Compute the mean IoU and Dice coefficient per class
meanIoU = totalIoU / numImages;
meanDice = totalDice / numImages;

% Compute Precision, Recall per class
precision = totalTP ./ (totalTP + totalFP);
recall = totalTP ./ (totalTP + totalFN);

% Display the results in a table
resultsTable = table(classNames', meanIoU, meanDice, precision, recall, totalTP, totalFP, totalFN, totalTN, 'VariableNames', {'Class', 'Mean_IoU', 'Mean_Dice', 'Precision', 'Recall', 'Total_TP', 'Total_FP', 'Total_FN', 'Total_TN'});
disp(resultsTable);

% Compute the overall mean IoU and Dice coefficient across all classes
overallMeanIoU = mean(meanIoU);
overallMeanDice = mean(meanDice);
fprintf('Overall Mean IoU: %.4f\n', overallMeanIoU);
fprintf('Overall Mean Dice: %.4f\n', overallMeanDice);

%% Calculate the global accuracy
totalTPAllClasses = sum(totalTP);
totalFPAllClasses = sum(totalFP);
totalFNAllClasses = sum(totalFN);
totalTNAllClasses = sum(totalTN);

globalAccuracy = (totalTPAllClasses + totalTNAllClasses) / ...
                 (totalTPAllClasses + totalFPAllClasses + totalFNAllClasses + totalTNAllClasses);

fprintf('Global Accuracy: %.4f\n', globalAccuracy);

%% Visualise an image with the model for qualitative
% Read an image from the test set
testImageIdx = 1;  % Index of the test image to visualize
I = readimage(imdsTest, testImageIdx);

% Perform semantic segmentation using the trained U-Net model
C = semanticseg(I, net);

% Read the corresponding ground truth label
expectedResult = readimage(pxdsTest, testImageIdx);

% Display the original image
figure;
subplot(1, 3, 1);
imshow(I);
title('Original Image');

% Display the segmented image
subplot(1, 3, 2);
imshow(labeloverlay(I, C, 'Colormap', cmap, 'Transparency', 0.4));
title('Segmented Image');

% Display the ground truth label
subplot(1, 3, 3);
imshow(labeloverlay(I, expectedResult, 'Colormap', cmap, 'Transparency', 0.4));
title('Ground Truth Labels');