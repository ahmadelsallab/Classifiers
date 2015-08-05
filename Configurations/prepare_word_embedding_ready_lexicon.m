
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Ready lexicon vocabulary %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vocabFileName = '..\..\..\..\OMA\Code\Datasets\ArSenL\vocabulary.txt';
scoresFileName = '..\..\..\..\OMA\Code\Datasets\ArSenL\ArSenL_average_BW.txt';


fid = fopen(vocabFileName,'r','n','UTF-8');
words = {};
line = fgets(fid);
num = 1;
while (line > 0)
    fprintf(1, 'Reading word number %d\n', num);
    words = [words; line];
    line = fgets(fid);
    num = num + 1;
end
words = unique(words);
wordMap = containers.Map(words,1:length(words));
vocab_size = length(words);

save(['input_data_We_' num2str(ngram) '.mat']);
save('vocab_ArSenL_We.mat', 'words');

%%%%%%%%%%%%%%%%%%%%%%%%%% Load ready indices %%%%%%%%%%%%%%%%%%%%%%%%%%%
txtFileName = '..\..\..\..\OMA\Code\Datasets\\ArSenL\corpus lemmas.txt';
indicesFileName = '..\..\..\..\OMA\Code\Datasets\ArSenL\indices.txt';
fid = fopen(txtFileName,'r','n','UTF-8');
indices = csvread(indicesFileName);
mFeatures = []; % Ncases x ngram
mTargets = []; % valid = [0 1;] invalid = [1 0];
line = fgets(fid);
num = 1;
while line > 0    
    fprintf(1, 'Reading line number %d\n', num);
    line = strtrim(line);
    non_zero = find(indices(num,:) == 0);
    non_zero = non_zero(1);
    lineWordsMat = indices(num, 1 : non_zero - 1);    
    
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
    
    num = num + 1;
    line = fgets(fid);
end



% Save the workspace
save(['input_data_We_' num2str(ngram) '.mat']);
% Close read and write files
fclose(fid);


