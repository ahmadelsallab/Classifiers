%clear, clc;
%indices = '..\..\..\..\OMA\Code\Datasets\ArSenL\indices.txt';
%vocabulary = '..\..\..\..\OMA\Code\Datasets\ArSenL\vocabulary.txt';
%delimiter = ';';
vocabulary = '..\..\..\..\OMA\Code\Datasets\IMDB\IMDB2000ExtendedLexicon';
delimiter = '	';

%corpus = '..\..\..\..\OMA\Code\Datasets\ArSenL\corpus lemmas.txt';
global CONFIG_strParamsGUI;
if(~isempty(CONFIG_strParamsGUI))
    vocabulary = CONFIG_strParamsGUI.sVocabularyFilePath;
    %indices = CONFIG_strParamsGUI.sIndicesFilePath;
end
% Get the scores of words
fid_voc = fopen(vocabulary,'r','n','UTF-8');

line = fgets(fid_voc);

words = {};
mFeatures = [];
mTargets = [];
num = 1;
while (line > 0)
    
    %line = strtrim(line);

    lineScoresSplit = textscan(line,'%s','delimiter',delimiter);

    fprintf(1, 'Reading line %d\n', num);    

    words = [words; lineScoresSplit{1}{1}];

    mFeatures = [mFeatures; num];
    global bLexiconEmbeddingObjectiveScoreIncluded;
    if(bLexiconEmbeddingObjectiveScoreIncluded == 1)
        
        % With objective
        mTargets = [mTargets; [str2double(lineScoresSplit{1}{2}) str2double(lineScoresSplit{1}{3}) str2double(lineScoresSplit{1}{4})];];
    else
        % No objective, just +/-
        mTargets = [mTargets; [str2double(lineScoresSplit{1}{2}) str2double(lineScoresSplit{1}{3})];];
    end

    % Get next line
    line = fgets(fid_voc);
    num = num + 1;
end
mTargets = mTargets * 100;
vocab_size = length(words);

save('vocab_SentiWordNet_Embedding.mat', 'words');
save(['input_data_SentiWordNet_Embedding_' num2str(ngram) '.mat']);

fclose(fid_voc);