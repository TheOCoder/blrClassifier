%BLRCLASSIFIER Bayesian Linear Regression Classifier object
%   This is version 2014.01.26 - check for updates at:
%   https://github.com/TheOCoder/blrClassifier/
%   
%   The update equations used in EM iterations are based on:
%   P.-J. Kindermans, H. Verschore, and B. Schrauwen “A Unified Probabilistic Approach to Improve Spelling in an Event-related Potential Based Brain-computer Interface.”, Ieee Transactions on Biomedical Engineering, 2013
%   
%   
%   CObj = blrClassifier creates a BLR classifier with default settings
%   CObj = blrClassifier(settings) creates a BLR classifier with the given settings
%   'settings' is a struct of settings which can contain some of the following fields. 
%   Values in brackets [] are the default value of the options and are employed if such field is not available in 'settings'
%   Any option not specified in 'settings' will be set to the default value
%      .addBiasTerm [1] -> Add a bias term (a feature which is always "1" to the data). This is highly recommended!
%      .silent [1] -> if silent = 0 then many actions will be reported in the Matlab console (good for debugging)
%      .noEMIterations [0] -> if you want no EM iterations over the parameters during training (can sometimes be helpful, for example when alpha or beta are not converging for your dataset)
%      .EMIterations [0] -> if you want to do exactly a special number of EM iterations [0 means do EM until convergance or MaxEMIterations]
%      .MaxEMIterations [300] -> Max number of EMs (to avoid unconvergance problems)
%      .convergenceEpsilon [1e-3] -> Convergance threshold
%      .betaMax [30] -> Maximum possible beta (0 means beta has no maximum)
%      .betaMin [0] -> Minimum possible beta (0 means beta has no minimum)
%      .alphaMax [0] -> Maximum possible alpha (0 means alpha has no maximum)
%      .alphaMin [1e-6] -> Minimum possible alpha (0 means alpha has no minimum)
%      .priors [0] -> If you want to set class priors manually (0 means class priors should be determined from relative frequency in train data)
%
%   Example:
%      settings = struct();
%      settings.addBiasTerm = 0; % The default for addBiasTerm is 1 but
%      %    here we don't want the classifier to add bias term (which is a bad
%      %    idea!) so we set addBiasTerm to 0
%      CObj = blrClassifier(settings);
%      % Now CObj is ready to use
%   
%   Once a blrClassifier object (such as CObj in the above example) is
%   available, one can use methods such as train and classify to use it
%   Example:
%      trainData = X; % A matrix containing training dataset - Each row
%      %    must be a trainig sample and each column contains a feature value
%      trainTarget = T; % A column vector containing the correct labels for
%      %    the training data - number of elements is supposed to be the same as
%      %    trainData rows - Class labels are expected to be -1 and 1
%      CObj.train(trainData, trainTarget);
%      
%      testData = X; % A matrix containing test dataset - Num of columns (features)
%      %    must be equal to number of trainData columns
%      probabilities = CObj.classify(testData);
%      %    probabilities will be a column vector with the same number of
%      %    rows as the testData and each element will contain a number
%      %    between 0 and 1 -> the probability that the class of the corresponding
%      %    testData is '1'
%   
%   
%   Notice:
%      CObj2 = CObj; 
%      % This will NOT copy the classifier, but CObj2 will point to the same classifier as CObj
%      
%      % In order to enable the copy functionality, set the first line of
%      % code in blrClassifier.m to the following:
%      classdef blrClassifier < matlab.mixin.Copyable
%      
%      % This is not the difault because some MATLAB versions don't support it
%   
%   
%   

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bayesian Linear Regression Classifier Class
% by Omid Ghasemsani - omidsani@gmail.com
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% classdef blrClassifier < matlab.mixin.Copyable % used instead of handle to enable copy function
classdef blrClassifier < handle
    properties
        settings
        classes
        classLabels
        numOfFeatures 
        w    % features x 1 matrix
        beta % is 1/sigma
        alpha
        muW
        latestLoglikelihood
    end
    methods
        function obj = blrClassifier(varargin)
            % 'settings' is a struct of settings which can contain some of the following fields. 
            % Values in brackets are the default value of the options and are employed if such field is not available in 'settings'
                % .addBiasTerm [1] -> Add a bias term (a feature which is always "1" to the data). This is highly recommended!
                % .silent [1] -> if silent = 0 then many actions will be reported in the Matlab console (good for debugging)
                % .noEMIterations [0] -> if you want no EM iterations over the parameters during training (can sometimes be helpful, for example when alpha or beta are not converging for your dataset)
                % .EMIterations [0] -> if you want to do exactly a special number of EM iterations [0 means do EM until convergance or MaxEMIterations]
                % .MaxEMIterations [300] -> Max number of EMs (to avoid unconvergance problems)
                % .convergenceEpsilon [1e-3] -> Convergance threshold
                % .betaMax [30] -> Maximum possible beta (0 means beta has no maximum)
                % .betaMin [0] -> Minimum possible beta (0 means beta has no minimum)
                % .alphaMax [0] -> Maximum possible alpha (0 means alpha has no maximum)
                % .alphaMin [1e-6] -> Minimum possible alpha (0 means alpha has no minimum)
                % .priors [0] -> If you want to set class priors manually (0 means class priors should be determined from relative frequency in train data)

            % Defaults
            obj.settings = struct();
            if nargin > 0, obj.settings = varargin{1}; end

            % processing settings and setting default values    
            if (~isstruct(obj.settings)), error('ERROR: settings is not give as a struct.\n'); end
            if (~isfield(obj.settings, 'addBiasTerm')), obj.settings.addBiasTerm = 1; end
            if (~isfield(obj.settings, 'silent')), obj.settings.silent = 0; end
            
            if (~isfield(obj.settings, 'noEMIterations')), obj.settings.noEMIterations = 0; end
            if (~isfield(obj.settings, 'EMIterations')), obj.settings.EMIterations = 00; end
            if (~isfield(obj.settings, 'MaxEMIterations')), obj.settings.MaxEMIterations = 300; end
            if (~isfield(obj.settings, 'convergenceEpsilon')), obj.settings.convergenceEpsilon = 1e-3; end
            if (~isfield(obj.settings, 'betaMax')), obj.settings.betaMax = 30; end % 0 means beta has no maximum
            if (~isfield(obj.settings, 'betaMin')), obj.settings.betaMin = 0; end % 0 means beta has no minimum
            if (~isfield(obj.settings, 'alphaMax')), obj.settings.alphaMax = 0; end % 0 means alpha has no maximum
            if (~isfield(obj.settings, 'alphaMin')), obj.settings.alphaMin = 1e-6; end % 0 means alpha has no minimum
            if (~isfield(obj.settings, 'priors')), obj.settings.priors = 0; end
            % These settings are not supposed to be changed! (Some are totally useless, and some are under developement)
            if (~isfield(obj.settings, 'storeTrainSignal')), obj.settings.storeTrainSignal = 0; end
            if (~isfield(obj.settings, 'alpha')), obj.settings.alpha = 1; end
            if (~isfield(obj.settings, 'beta')), obj.settings.beta = 1; end
            if (~isfield(obj.settings, 'weights')), obj.settings.weights = 0; end
            if (~isfield(obj.settings, 'muWeights')), obj.settings.muWeights = 0; end
            if (~isfield(obj.settings, 'numOfFeatures')), obj.settings.numOfFeatures = 0; end            
        end
        function initialize(obj)
            obj.numOfFeatures = obj.settings.numOfFeatures;
            if (obj.numOfFeatures < 1), error('ERROR: Classifier cannot be initialized with zero features;\n'); end
            obj.alpha = obj.settings.alpha;
            obj.beta = obj.settings.beta;
            
            if (obj.alpha < obj.settings.alphaMin), obj.alpha = obj.settings.alphaMin; end
            if (obj.settings.alphaMax > 0 && obj.alpha > obj.settings.alphaMax), obj.alpha = obj.settings.alphaMax; end
            
            if (obj.settings.muWeights == 0) % Mu of Weights not given
                obj.muW = zeros(obj.numOfFeatures, 1); % features x 1 matrix
            else
                if (length(obj.settings.muWeights) ~= obj.numOfFeatures), error('ERROR: muWeights given do not match number of features\n'); end
                obj.muW = obj.settings.muWeights;
            end
            
            sigmaW = eye(obj.numOfFeatures) / obj.alpha; % while initializing we assume alpha = 1 % TO DO: make sure it is correct (not  * obj.alpha)
            
            if (obj.settings.weights == 0) % Weights not given
                obj.w = mvnrnd(obj.muW, sigmaW)';
            else
                if (length(obj.settings.weights) ~= obj.numOfFeatures), error('ERROR: weights given do not match number of features\n'); end
                obj.w = obj.settings.weights;
            end
            obj.latestLoglikelihood = -Inf;
            
            obj.alpha = obj.numOfFeatures / ( (obj.w - obj.muW)' * (obj.w - obj.muW) );
            if (obj.alpha < obj.settings.alphaMin), obj.alpha = obj.settings.alphaMin; end
            if (obj.settings.alphaMax > 0 && obj.alpha > obj.settings.alphaMax), obj.alpha = obj.settings.alphaMax; end
            
            obj.classes{1}.init = 'test';
            obj.classes{2}.init = 'test';
        end
        function train(obj, data, varargin)
            if (obj.settings.addBiasTerm)                
                % add bias feature to the data
                data = cat(2, data, ones(size(data, 1), 1));
            end
            if (obj.settings.numOfFeatures == 0) % Set the number of feature if it has not been set yet
                obj.settings.numOfFeatures = size(data, 2);
                obj.initialize();
            end            
            
            samples = size(data, 1);
            if (size(data, 2) ~= obj.numOfFeatures), error('ERROR: Features in the data are not consistent with featureas expected by classifier!\n'); end
            
            if nargin > 2  % We have labels -> supervised training of weight vector
                labels = varargin{1}; 
                if (size(labels, 1) ~= samples), error('ERROR: Class labels given doesnt match the data samples!\n'); end
                if (size(labels, 2) ~= 1), error('ERROR: Class labels must be given given as a vertical vector!\n'); end
                
                obj.classLabels = unique(labels)';
                if (min(obj.classLabels == [-1 1]) < 1), error('ERROR: Class labels are expected to be -1 and 1. And at least one sample of each class must be given in training.\n'); end
                
                if ( (min(size(obj.settings.priors) == [2 1]) == 1) && (sum(obj.settings.priors)==1) ) % prior values are given by caller
                    priors = obj.settings.priors;
                    fprintf('Priors for classes were set to %.2f\n', priors);
                    fprintf('set "priors" setting to 0 if you want the classifier to use relative probability of classes in the train data\n');
                end
            
                X = data;
                T = labels;
                
                XtransMultipliedByX = X' * X;
                D = obj.numOfFeatures;
                
                
                if (obj.settings.noEMIterations)
                    % Calculate proper weights (w) without changing alpha
                    % and beta
                    Ainv = obj.beta * XtransMultipliedByX + obj.alpha * eye(D);
                    A = inv(Ainv);
                    obj.w = obj.beta * A * X' * T;
                    Y = X * obj.w; % data mapped to one dimention
                else
                    overfittingRisk = 0;
                    iter = 0;
                    evaluationDone = 0;

                    while (~evaluationDone)
                        iter = iter+1;

                        % Calculate proper weights (w)
                        Ainv = obj.beta * XtransMultipliedByX + obj.alpha * eye(D);
                        A = inv(Ainv);
                        obj.w = obj.beta * A * X' * T;

                        Y = X * obj.w; % data mapped to one dimention

                        alphaNew = D / ( obj.w' * obj.w + trace(A) );
    %                     betaNew = samples / ( trace( Ainv * XtransMultipliedByX ) + (Y-T)' * (Y-T) ); % This method wont converge!
                        betaNew = samples / ( (Y-T)' * (Y-T) ); % This method converges
    %                     betaNew = ( std(Y(T == +1)) * sum(T == +1) + std(Y(T == -1)) * sum(T == -1) ) / length(Y); % This method has no mathematical BG And wont converge

                        betaChange = abs( abs(betaNew - obj.beta) / obj.beta );
                        alphaChange = abs( abs(alphaNew - obj.alpha) / obj.alpha );

                        obj.beta = betaNew;
                        obj.alpha = alphaNew;

                        % Check for convergence
                        if ( (obj.settings.convergenceEpsilon > 0) && (alphaChange + betaChange < obj.settings.convergenceEpsilon )), 
                            evaluationDone = 1; 
                            if (~obj.settings.silent), fprintf('EM converged at '); end
                        end
                        if (obj.settings.MaxEMIterations <= iter), 
                            evaluationDone = 1; 
                            if (~obj.settings.silent), fprintf('Max number of allowed EM iterations reached at '); end
                        end
                        if ( (obj.settings.EMIterations > 0) && (obj.settings.EMIterations <= iter)), 
                            evaluationDone = 1; 
                            if (~obj.settings.silent), fprintf('Requested EM iterations finished at '); end
                        end

                        if (~obj.settings.silent), fprintf('Iter %d:  Alpha = %.4f  -  Beta = %.4f \n', iter, obj.alpha, obj.beta); end
                        if ( (obj.settings.alphaMax > 0) && (obj.alpha > obj.settings.alphaMax)), % If we have a maximum limit for alpha and it has been crossed
                            fprintf('WARNING: Alpha was set to its maximum = %.4f (from %.4f)\n', obj.settings.alphaMax, obj.alpha);
                            obj.alpha = obj.settings.alphaMax;
                        end
                        if ( (obj.settings.alphaMin > 0) && (obj.alpha < obj.settings.alphaMin)), % If we have a minimum limit for alpha and it has been crossed
                            fprintf('WARNING: Alpha was set to its minimum = %.4f (from %.4f)\n', obj.settings.alphaMin, obj.alpha);
                            obj.alpha = obj.settings.alphaMin;
                        end
                        if ( (obj.settings.betaMax > 0) && (obj.beta > obj.settings.betaMax)), % If we have a maximum limit for beta and it has been crossed
                            fprintf('WARNING: Beta was set to its maximum = %.4f (from %.4f) - Overfitting Risk is high\n', obj.settings.betaMax, obj.beta);
                            obj.beta = obj.settings.betaMax;
                            overfittingRisk = overfittingRisk + 1;
                        end
                        if ( (obj.settings.betaMin > 0) && (obj.beta < obj.settings.betaMin)), % If we have a minimum limit for beta and it has been crossed
                            fprintf('WARNING: Beta was set to its minimum = %.4f (from %.4f)\n', obj.settings.betaMin, obj.beta);
                            obj.beta = obj.settings.betaMin;
                        end

                        if (overfittingRisk > 5)
                            fprintf('WARNING: Beta and/or Alpha repeatedly crossed their limits. EM is unlikely to converge!\n');
                            evaluationDone = 1;
                        end
                    end
                    
                    if (overfittingRisk > 0)
                        fprintf('To avoid over fitting try using less features or more training samples\n');
                        fprintf('As a rule of thumb, the number of training samples (now %d) should be more than the number of features (now %d)\n', samples, D);
                        fprintf('Or to get a slightly better result you can disable EM iterations altoghether by setting "noEMIterations" to 1\n');
                    end   
                
                end
                
                sigma = cov(Y); % Variance of all data mapped into one direction
                
                numOfClasses = 0;
                for c = obj.classLabels
                    numOfClasses = numOfClasses + 1;
                    classData = X((T==c), :);
                    Yc = Y((T==c), :); % class data mapped to one dimention

                    numOfClassSamples = size(Yc, 1);
                    
                    if (obj.settings.priors), % Guess priors
                        prior = priors(numOfClasses);
                    else
                        prior = size(Yc, 1) / size(Y, 1);
                    end
                    
                    mu = mean(Yc);
                    inClassSigma = cov(Yc);

                    obj.classes{numOfClasses} = struct('mu', mu, 'sigma', sigma, 'inClassSigma', inClassSigma, 'label', c, 'prior', prior, 'numOfClassSamples', numOfClassSamples);
                    if (obj.settings.storeTrainSignal)
                        obj.classes{numOfClasses}.classSamples = classData;
                    end
                end
                
                % This is a workaround
                obj.beta = 1 / ( obj.classes{1}.prior * obj.classes{1}.inClassSigma + obj.classes{2}.prior * obj.classes{2}.inClassSigma );
%                 obj.beta = 1 / sigma;                
%                 obj.alpha = obj.numOfFeatures / ( obj.w' * obj.w );
                
                
                if (obj.settings.alphaMax > 0 && obj.alpha > obj.settings.alphaMax)
                    fprintf('WARINING: alpha > alphaMax\n');
                end
                
                if (~obj.settings.silent), fprintf('Training result --> means: nonP300 -> %.4f  | P300 -> %.4f\n', obj.classes{1}.mu, obj.classes{2}.mu); end
            end
        end
        function probabilities = classify(obj, data, varargin)
            if (obj.settings.addBiasTerm)                
                % add bias feature to the data
                data = cat(2, data, ones(size(data, 1), 1));
            end
            
            samples = size(data, 1);
            if (size(data, 2) ~= obj.numOfFeatures), error('ERROR: Features in the data are not consistent with featureas expected by classifier!\n'); end
            
            trueClassesAvailable = 0;
            trueClasses = ones(size(data, 1), 1);
            
            if nargin > 2, 
                trueClassesAvailable = 1;
                trueClasses = varargin{1}; 
            end

            if (samples ~= length(trueClasses)), error('ERROR: trueClasses given doesnt match the data samples!\n'); end

            X = data;
            
%             numOfClasses = length(obj.classLabels); % Not for unsupervised
            numOfClasses = 2;
            probs = zeros(samples, numOfClasses);
            
            Y = X * obj.w;
%             figure;p = linspace(-3,3,100);o1 = normpdf(p,obj.classes{1}.mu, 1 / obj.beta);o2 = normpdf(p,obj.classes{2}.mu, 1 / obj.beta);plot(p,o1,p,o2,p,o1./(o1+o2),p,o2./(o1+o2));
%             figure;p = linspace(-3,3,100);o1 = obj.classes{1}.prior * normpdf(p,obj.classes{1}.mu, 1 / obj.beta);o2 = obj.classes{2}.prior * normpdf(p,obj.classes{2}.mu, 1 / obj.beta);plot(p,o1,p,o2,p,o1./(o1+o2),p,o2./(o1+o2));
            for i = 1:numOfClasses
                probs(:,i) = obj.classes{i}.prior * normpdf(Y, obj.classes{i}.mu, 1 / obj.beta);
            end
            
            probsCopy = probs;
            probs(:,1) = probsCopy(:,1) ./ (probsCopy(:,1)+probsCopy(:,2));
            probs(:,2) = probsCopy(:,2) ./ (probsCopy(:,1)+probsCopy(:,2));
            
            probs(isnan(probs)) = 0; % TO DO: do something more logical!
            
%             [probabilities probableClasses] = max(probs, [], 2);
            
            probabilities = probs;
            
%             labels = obj.classLabels(probableClasses(:))';

%             corrects = samples - sum(trueClasses ~= labels);
%             if (trueClassesAvailable)
%                 fprintf('Classification performance: %.2f (%d/%d)\n', corrects/samples*100, corrects, samples);
%             else
%                 fprintf('Classification performance: not available (true classes not given) [%.2f (%d/%d)]\n', corrects/samples*100, corrects, samples);
%             end
        end
    end
end
