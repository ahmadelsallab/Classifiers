
%clear, clc;
covered_in_arsenl = '..\..\..\..\OMA\Code\Datasets\ArSenL\covered in arsenl.txt';
sentence_vectors = '..\..\..\..\OMA\Code\Datasets\ArSenL\sentence vectors.txt';
corpus4rae = '..\..\..\..\OMA\Code\Datasets\ArSenL\corpus4rae.txt';
closed_class = '..\..\..\..\OMA\Code\Datasets\ArSenL\closed-class.txt';
punctuations = '..\..\..\..\OMA\Code\Datasets\ArSenL\punctuations.txt';
sentence_medians = '..\..\..\..\OMA\Code\Datasets\ArSenL\sentence medians.txt';

% Get the scores of words
fid = fopen(corpus4rae,'r','n','UTF-8');
fid_sentence_vec = fopen(sentence_vectors,'r','n','UTF-8');
fid_sentence_medians = fopen(sentence_medians,'r','n','UTF-8');
fid_punc = fopen(punctuations,'r','n','UTF-8');
fid_covered_in_arsenl = fopen(covered_in_arsenl,'r','n','UTF-8');
fid_closed_class = fopen(closed_class,'r','n','UTF-8');

% 1. Get all punctuation words
% Form wordmap of punc. words
line_punc = fgets(fid_punc);
words_punc = {};
while(line_punc > 0)
    line_punc = strtrim(line_punc);
    words_punc = [words_punc; line_punc];
    line_punc = fgets(fid_punc);
end
words_punc = unique(words_punc);
puncWordMap = containers.Map(words_punc,1:length(words_punc));

% 2. Get all covered in arsenl words
% Form wordmap of covered in arsenl words
line_arsenl = fgets(fid_covered_in_arsenl);
words_arsenl = {};
while(line_arsenl > 0)
    line_arsenl = strtrim(line_arsenl);
    words_arsenl = [words_arsenl; line_arsenl];
    line_arsenl = fgets(fid_covered_in_arsenl);
end
words_arsenl = unique(words_arsenl);
coveredInArsenlWordMap = containers.Map(words_arsenl,1:length(words_arsenl));

% 3. Get all closed-class words
% Form wordmap of closed-class words
line_closed_class = fgets(fid_closed_class);
words_closed_class = {};
while(line_closed_class > 0)
    line_closed_class = strtrim(line_closed_class);
    words_closed_class = [words_closed_class; line_closed_class];
    line_closed_class = fgets(fid_closed_class);
end
words_closed_class = unique(words_closed_class);
closedClassWordMap = containers.Map(words_closed_class,1:length(words_closed_class));

line = fgets(fid);
line_scores = fgets(fid_sentence_vec);
line_sentence_medians = fgets(fid_sentence_medians);
wordsMap = containers.Map();
words = {};
mFeatures = [];
mTargets = [];
wordNum = 1;
num = 1;
while (line > 0)
    
    line = strtrim(line);
    line_scores = strtrim(line_scores);
    line_sentence_medians = strtrim(line_sentence_medians);
    lineScoresSplit = textscan(line_scores,'%s','delimiter',',');
    lineSentenceMediansSplit = textscan(line_sentence_medians,'%s','delimiter',',');
    lineWords = splitLine(line);
    arsenl_sentence_vector_ptr = 1;

    fprintf(1, 'Reading line %d\n', num);
    
    for wordIdx = 1 : length(lineWords)
        % If new word
        if ~wordsMap.isKey(lineWords{wordIdx})
        % For every sentence in (corpus4rae.txt) scan the sentence word by word:
        % ​If current word belongs to (punctuations.txt) or is a digit, then exclude it.
        % If current word belongs to (covered in arsenl.txt) then extract the triplet of scores from (sentence_vectors.txt).
        % If current word belongs to (closed-class.txt) then assign to it the neutral score triplet (0,0,1)
        % If current word does not belong to any of these lists, then assign to it the median triplet found in (sentence medians.txt) in the line corresponding to the sentence at hand.
            if(puncWordMap.isKey(lineWords{wordIdx}))
                continue;
            elseif(coveredInArsenlWordMap.isKey(lineWords{wordIdx}))
                % Add the word
                words = [words; lineWords{wordIdx}];
                % Add the scores
                wordsMap(lineWords{wordIdx}) = [str2num(lineScoresSplit{1}{arsenl_sentence_vector_ptr}) str2num(lineScoresSplit{1}{arsenl_sentence_vector_ptr + 1}) str2num(lineScoresSplit{1}{arsenl_sentence_vector_ptr + 2})];
                arsenl_sentence_vector_ptr = arsenl_sentence_vector_ptr + 3;
                
                mFeatures = [mFeatures; wordNum];
                mTargets = [mTargets; wordsMap(lineWords{wordIdx})];
                wordNum = wordNum + 1;
            elseif(closedClassWordMap.isKey(lineWords{wordIdx}))
                % Add the word
                words = [words; lineWords{wordIdx}];
                wordsMap(lineWords{wordIdx}) = [0 0 1];
                
                mFeatures = [mFeatures; wordNum];
                mTargets = [mTargets; wordsMap(lineWords{wordIdx})];
                wordNum = wordNum + 1;
            else
                % Add the word
                words = [words; lineWords{wordIdx}];
                wordsMap(lineWords{wordIdx}) = [str2num(lineSentenceMediansSplit{1}{1}) str2num(lineSentenceMediansSplit{1}{2}) str2num(lineSentenceMediansSplit{1}{3})];
                
                mFeatures = [mFeatures; wordNum];
                mTargets = [mTargets; wordsMap(lineWords{wordIdx})];
                wordNum = wordNum + 1;
            end
            
        end
    end
    % Get next line
    line = fgets(fid);
    line_scores = fgets(fid_sentence_vec);
    line_sentence_medians = fgets(fid_sentence_medians);
    
    num = num + 1;
end
vocab_size = length(words);

save('vocab_ArSenL_We.mat', 'words');
save(['input_data_ArSenL_We_' num2str(ngram) '.mat']);

fclose(fid);
fclose(fid_sentence_vec);
fclose(fid_sentence_medians);
fclose(fid_punc);
fclose(fid_covered_in_arsenl);
fclose(fid_closed_class);