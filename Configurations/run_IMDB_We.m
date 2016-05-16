clear, clc, close all;


ngram  = 2;
preProFile = ['input_data_We_' num2str(ngram) '.mat'];
prepare_IMDB_word_embedding;
    

fprintf(1, 'Configuring...\n');
% Start Configuration
[CONFIG_strParams] = CONFIG_setConfigParams_IMDB_We();

fprintf(1, 'Configuration done successfuly\n');

% Change directory to go there
cd(CONFIG_strParams.sDefaultClassifierPath);

% Call main entry function of the classifier
MAIN_trainAndClassify(CONFIG_strParams);