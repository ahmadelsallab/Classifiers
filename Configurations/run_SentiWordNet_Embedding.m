clear, clc, close all;

fprintf(1, 'Configuring...\n');
ngram  = 1;
preProFile = ['input_data_SentiWordNet_Embedding_' num2str(ngram) '.mat'];
global bLexiconEmbeddingObjectiveScoreIncluded;
bLexiconEmbeddingObjectiveScoreIncluded = 0;
global CONFIG_strParamsGUI;
if(~isempty(CONFIG_strParamsGUI))
    bLexiconEmbeddingObjectiveScoreIncluded = CONFIG_strParamsGUI.bLexiconEmbeddingObjectiveScoreIncluded;
end
% read in polarity dataset
prepare_word_SentiWordNet_embedding_Ready;

    


% Start Configuration
[CONFIG_strParams] = CONFIG_setConfigParams_SentiWordNet_Embedding();

fprintf(1, 'Configuration done successfuly\n');

% Change directory to go there
cd(CONFIG_strParams.sDefaultClassifierPath);

% Call main entry function of the classifier
MAIN_trainAndClassify(CONFIG_strParams);