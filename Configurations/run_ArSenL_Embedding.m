clear, clc, close all;

fprintf(1, 'Configuring...\n');
ngram  = 1;
preProFile = ['input_data_ArSenL_Embedding_' num2str(ngram) '.mat'];
global bLexiconEmbeddingObjectiveScoreIncluded;
bLexiconEmbeddingObjectiveScoreIncluded = 0;
% read in polarity dataset
if ~exist(preProFile,'file')
    %prepare_word_ArSenL_embedding;
    prepare_word_ArSenL_embedding_Ready;
end
    


% Start Configuration
[CONFIG_strParams] = CONFIG_setConfigParams_ArSenL_Embedding();

fprintf(1, 'Configuration done successfuly\n');

% Change directory to go there
cd(CONFIG_strParams.sDefaultClassifierPath);

% Call main entry function of the classifier
MAIN_trainAndClassify(CONFIG_strParams);