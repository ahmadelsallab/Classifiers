% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program reads raw MNIST files available at 
% http://yann.lecun.com/exdb/mnist/ 
% and converts them to files in matlab format 
% Before using this program you first need to download files:
% train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz 
% t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
% and gunzip them. You need to allocate some space for this.  

% This program was originally written by Yee Whye Teh

% Function: Modified from Ruslan Salakhutdinov and Geoff Hinton (See copyright above) 
% Converts the input NIST files into train and test features and mTrainmTestTargets vector
% Inputs:
% CONFIG_strParams: Reference to the configurations parameters structure
% Output:
% mTrainFeatures, mTrainmTrainmTestTargets, mTestFeatures, mTestmTrainmTestTargets save into CONFIG_strParams.sInputDataWorkspace
function DCONV_convertATB_Senti(CONFIG_strParams)

	features_file_name = [CONFIG_strParams.sDatasetFilesPath CONFIG_strParams.sFeaturesFileName];
    targets_file_name = [CONFIG_strParams.sDatasetFilesPath 'annotation_sentiment.txt'];
    f = csvread(features_file_name);
    t = csvread(targets_file_name);

    % Split into train and test
    mTrainFeatures = f(237:end,:);
    mTestFeatures = f(1:236,:);
    t_train = t(237:end,:);
    mTrainTargets = zeros(size(t_train, 1), 2);
    for i = 1 : size(t_train, 1)
       mTrainTargets(i, t_train(i)) = 1;
    end

    t_test = t(237:end,:);
    mTestTargets = zeros(size(t_test, 1), 2);
    for i = 1 : size(t_test, 1)
       mTestTargets(i, t_test(i)) = 1;
    end
    
    save(CONFIG_strParams.sInputDataWorkspace, '-v7.3', 'mTestFeatures', 'mTestTargets', 'mTrainFeatures', 'mTrainTargets');
end % end function
