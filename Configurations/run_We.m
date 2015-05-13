clear, clc, close all;

fprintf(1, 'Configuring...\n');
ngram  = 5;
preProFile = ['input_data_We_' num2str(ngram) '.mat'];

% read in polarity dataset
if ~exist(preProFile,'file')
    prepare_word_embedding;
end
    


% Start Configuration
[CONFIG_strParams] = CONFIG_setConfigParams_We();

fprintf(1, 'Configuration done successfuly\n');

% Change directory to go there
cd(CONFIG_strParams.sDefaultClassifierPath);

% Call main entry function of the classifier
MAIN_trainAndClassify(CONFIG_strParams);