
%clear, clc;
%indices = '..\..\..\..\OMA\Code\Datasets\ArSenL\indices.txt';
vocabulary = '..\..\..\..\OMA\Code\Datasets\ArSenL\vocabulary.txt';
%corpus = '..\..\..\..\OMA\Code\Datasets\ArSenL\corpus lemmas.txt';

% Get the scores of words
fid_voc = fopen(vocabulary,'r','n','UTF-8');

line = fgets(fid_voc);

words = {};
mFeatures = [];
mTargets = [];
num = 1;
while (line > 0)
    
    %line = strtrim(line);

    lineScoresSplit = textscan(line,'%s','delimiter',',');

    fprintf(1, 'Reading line %d\n', num);    

    words = [words; lineScoresSplit{1}{1}];

    mFeatures = [mFeatures; num];
    % With objective
    %mTargets = [mTargets; [str2double(lineScoresSplit{1}{2}) str2double(lineScoresSplit{1}{3}) str2double(lineScoresSplit{1}{4})];];
    % No objective, just +/-
    mTargets = [mTargets; [str2double(lineScoresSplit{1}{2}) str2double(lineScoresSplit{1}{3})];];

    % Get next line
    line = fgets(fid_voc);
    num = num + 1;
end
vocab_size = length(words);

save('vocab_ArSenL_We.mat', 'words');
save(['input_data_ArSenL_We_' num2str(ngram) '.mat']);

fclose(fid_voc);