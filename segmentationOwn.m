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

%% Load images and labels
imds = imageDatastore(imageDir);
classNames = ["flower", "background"]; % Define the two classes
labelIDs   = {1, 3};  % Map class 1 to flower and 3 to background

%% Check if the toolbox is available and create datastore
assert(exist('pixelLabelDatastore', 'file') == 2, 'Image Processing Toolbox might be missing.');

pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);

%% Visualize an example
I = read(imds);  
C = read(pxds); 
C = C{1};        

figure;
imshowpair(I, labeloverlay(I, C), 'montage');


%% Update Datastores to Point to Respective Folders for training, testing and validation
imdsTrain = imageDatastore(trainImageDir);
imdsVal = imageDatastore(valImageDir);
imdsTest = imageDatastore(testImageDir);

pxdsTrain = pixelLabelDatastore(trainLabelDir, classNames, labelIDs);
pxdsVal = pixelLabelDatastore(valLabelDir, classNames, labelIDs);
pxdsTest = pixelLabelDatastore(testLabelDir, classNames, labelIDs);

%% Prepare the training and validation data
trainingData = pixelLabelImageDatastore(imdsTrain, pxdsTrain);
validationData = pixelLabelImageDatastore(imdsVal, pxdsVal);

%% Define the CNN architecture with correct upsampling
layers = [
    imageInputLayer([256 256 3])
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2) % Reduces to 128x128
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2) % Reduces to 64x64
    
    convolution2dLayer(3, 128, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2) % Reduces to 32x32

    convolution2dLayer(3, 256, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2) % Reduces to 16x16
    
    transposedConv2dLayer(4, 256, 'Stride', 2, 'Cropping', 'same') % Upsamples to 32x32
    reluLayer()
    
    transposedConv2dLayer(4, 128, 'Stride', 2, 'Cropping', 'same') % Upsamples to 64x64
    reluLayer()
    
    transposedConv2dLayer(4, 64, 'Stride', 2, 'Cropping', 'same') % Upsamples to 128x128
    reluLayer()
    
    transposedConv2dLayer(4, 32, 'Stride', 2, 'Cropping', 'same') % Upsamples to 256x256
    reluLayer()
    
    convolution2dLayer(1, numel(classNames))
    softmaxLayer()
    pixelClassificationLayer('Classes', classNames)
];

%% Visualise the architecture of custom model
lgraph = layerGraph(layers);
analyzeNetwork(lgraph);

%% Training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'VerboseFrequency', 2, ...
    'Plots', 'training-progress', ...
    'L2Regularization', 1e-4, ...
    'ValidationData', validationData, ...
    'ValidationFrequency', 10, ...
    'GradientThresholdMethod', 'l2norm', ...
    'GradientThreshold', 1);


%% Train the network
net = trainNetwork(trainingData, layers, options);

%% Save the trained network
save('segmentownnet.mat', 'net');

%% Load the trained network
load('segmentownnet.mat', 'net');

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
testImageIdx = 29;  % Index of the test image to visualize
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
