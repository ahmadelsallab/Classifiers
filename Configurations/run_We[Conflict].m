clear, clc, close all;


ngram  = 2;
preProFile = ['input_data_We_' num2str(ngram) '.mat'];
bReadyVocab = 1;
% read in polarity dataset
if ~exist(preProFile,'file')
    if(bReadyVocab == 1)
        prepare_word_embedding;
    else
        prepare_word_embedding_ready_lexicon;
    end
end
    

fprintf(1, 'Configuring...\n');
% Start Configuration
[CONFIG_strParams] = CONFIG_setConfigParams_We();

fprintf(1, 'Configuration done successfuly\n');

% Change directory to go there
cd(CONFIG_strParams.sDefaultClassifierPath);

% Call main entry function of the classifier
MAIN_trainAndClassify(CONFIG_strParams);