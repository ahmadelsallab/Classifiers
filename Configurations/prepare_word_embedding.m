%clear, clc;
txtFileName = '..\..\..\..\OMA\Code\Datasets\Qalb\Qalb compiled.txt';
global CONFIG_strParamsGUI;
if(~isempty(CONFIG_strParamsGUI))
    txtFileName = CONFIG_strParamsGUI.sUnsupervisedWeDatasetPath;
end
%ngram = 2;


% Open the file in UTF-8
fid_Qalb = fopen(txtFileName,'r','n','UTF-8');
% Get the sentences line by line
line = fgets(fid_Qalb);
%mFeatures = {};
words = {};
num = 1;
allSStr = {};

%%%%%%%%%%%%%%%%%% QALB VOC %%%%%%%%%%%%%%%%%%%%%%%
n_lines_max = 0;
global CONFIG_strParamsGUI;
if(~isempty(CONFIG_strParamsGUI))
    n_lines_max = CONFIG_strParamsGUI.nMaxNumLines;
end
% Load the positive and negative instances
% Save in the positive and negative separate txt files
% Save positive and negative cell arrays
% Build the vocabulary
while ((line > 0) & (num < n_lines_max))
    %mFeatures = [mFeatures; line];
    
    % Get the words of each line
    lineWords = splitLine(line);
    allSStr{num} = lineWords';
    words = [words; lineWords];
    num = num + 1;
    line = fgets(fid_Qalb);
    fprintf(1, 'Reading line number %d\n', num);
end

% Make unique vocabulary
words_Qalb = unique(words');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATB VOC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%txtFileName = '..\..\..\..\OMA\Code\Datasets\ATB\input\ATB1v3_UTF8.txt';
%annotationsFileName = '..\..\..\..\OMA\Code\Datasets\ATB\annotations.txt';
txtFileName = '..\..\..\..\OMA\Code\Datasets\ArSenL\corpus lemmas.txt';
annotationsFileName = '..\..\..\..\OMA\Code\Datasets\ArSenL\annotation_sentiment.txt';
global CONFIG_strParamsGUI;
if(~isempty(CONFIG_strParamsGUI))
    txtFileName = CONFIG_strParamsGUI.sSupervisedDataSetPath;
    annotationsFileName = CONFIG_strParamsGUI.sAnnotationsFilePath;
end

    fid_ATB = fopen(txtFileName,'r','n','UTF-8');
    labels = csvread(annotationsFileName);
        % Get the sentences line by line

    line = fgets(fid_ATB);
    %data = {};
    words = {};
    allSStr_pos = {};
    allSStr_neg = {};
    %num = 1;
    num_pos = 1;
    num_neg = 1;

    % Load the positive and negative instances
    % Save in the positive and negative separate txt files
    % Save positive and negative cell arrays
    % Build the vocabulary
    while line > 0        
        %data = [data; line];
        
        % Get the words of each line
        %lineWords = textscan(line,'%s','delimiter',' ');
        lineWords = splitLine(line);
        allSStr{num} = lineWords';
        words = [words; lineWords];
        num = num + 1;
        line = fgets(fid_ATB);
    end
    


    % Make unique vocabulary
    words_ATB = unique(words');
words = [words_Qalb words_ATB];
words = unique(words);
wordMap = containers.Map(words,1:length(words));
vocab_size = length(words);
save(['input_data_We_' num2str(ngram) '.mat']);
save('vocab_We.mat', 'words');
% Now score for each sentence the indices of words

allSNum = {};
mFeatures = []; % Ncases x ngram
mTargets = []; % valid = [0 1;] invalid = [1 0];
for lineIdx = 1 : size(allSStr, 2)
    lineWordsIndices = {};
    for wordIdx = 1 : size(allSStr{lineIdx}, 2)
        lineWordsIndices{wordIdx} = wordMap(allSStr{lineIdx}{wordIdx});
    end
    allSNum{lineIdx} = cell2mat(lineWordsIndices);
    lineWordsMat = allSNum{lineIdx};
    
    offset = 1;
    while (offset + ngram - 1) <= size(lineWordsMat, 2)
        % Create sets of ngram words indices
        % Label them all as valid
        valid_data = lineWordsMat(offset : offset + ngram - 1);        
        mTargets = [mTargets; [0 1];];
        mFeatures = [mFeatures; valid_data];
        
        % Replace random position by random vocabulary word
        random_vocab_word_idx = wordMap(words{randi(vocab_size,1,1)});
        invalid_data = valid_data;
        % Flip one index of each
        invalid_data(randi(ngram,1,1)) = random_vocab_word_idx;
        % Label the flipped ones as invalid
        mTargets = [mTargets; [1 0];];
        mFeatures = [mFeatures; invalid_data];
        
        offset = offset + ngram;
    end
    

    
end
% Save the workspace
save(['input_data_We_' num2str(ngram) '.mat']);

% Close read and write files
fclose(fid_Qalb);
fclose(fid_ATB);

