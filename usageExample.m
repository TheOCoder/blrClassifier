% Usage example for blrClassifier class
% The .blrClassifier folder must be in the MATLAB path

% Let's make a fake dataset
% we want to have 100 feature with a class specific average
numOfFeatures = 100;
numOfClass1Samples = 1000;
numOfClass2Samples = 1000;

% Prepare simulated data for the two classes
class1Average = 1;
class2Average = 2;
class1Data = class1Average + ones(numOfClass1Samples, numOfFeatures) .* randn(numOfClass1Samples, numOfFeatures);
class2Data = class2Average + ones(numOfClass1Samples, numOfFeatures) .* randn(numOfClass2Samples, numOfFeatures);

% Prepare labels for the two classes
class1Label = -1;
class2Label = +1;
class1Target = class1Label * ones(numOfClass1Samples, 1);
class2Target = class2Label * ones(numOfClass2Samples, 1);

% Visulizing first feature of train data
figure
plot([class1Data(:, 1) class2Data(:, 1)])
xlabel('Sample number');
ylabel('Value of feature number 1');
legend('Class 1', 'Class 2');

% Now let's creat a blrClassifier object
settings = struct(); % We want every setting to be set as diffault so settings will be an empty struct
blrClassifierObject = blrClassifier(settings);

% Now lets traing the classifier
trainData = [class1Data; class2Data]; % Vertically Concatenate the train data of the two classes
trainTarget = [class1Target; class2Target]; % Vertically Concatenate the target labels data of the two classes
blrClassifierObject.train(trainData, trainTarget);


% Now let's create some test data and classify them with the classifier
testDataAverages = [2; 2; 2; 1; 1; 1; 3; 4; 5; 0; -1; -2];
testData = repmat(testDataAverages, 1, numOfFeatures)  + ones(length(testDataAverages), numOfFeatures) .* randn(length(testDataAverages), numOfFeatures);
probabilities = blrClassifierObject.classify(testData);

% probabilities =
% 
%          0    1.0000
%          0    1.0000
%          0    1.0000
%     1.0000         0
%     1.0000    0.0000
%     1.0000         0
%     0.5000    0.5000
%     0.5000    0.5000
%     0.5000    0.5000
%     0.5000    0.5000
%     0.5000    0.5000
%     0.5000    0.5000
