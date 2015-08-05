

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATB VOC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vocabFileName = '..\..\..\..\OMA\Code\Datasets\ArSenL\ArSenL_Average_Arabic.txt';
scoresFileName = '..\..\..\..\OMA\Code\Datasets\ArSenL\ArSenL_average_BW.txt';


fid = fopen(vocabFileName,'r','n','UTF-8');
words = {};
line = fgets(fid);
while (line > 0)
    words = [words; line];
    line = fgets(fid);
end
words = unique(words);
wordMap = containers.Map(words,1:length(words));
vocab_size = length(words);
mFeatures = [];
for wordIdx = 1 : length(words)
    mFeatures = [mFeatures; wordMap(words{wordIdx})];
end

save('vocab_ArSenL_We.mat', 'words');

mTargets = [];
fclose(fid);
fid = fopen(scoresFileName,'r','n','UTF-8');

line = fgets(fid);
while (line > 0)
    lineSplit = textscan(line,'%s','delimiter',',');
    mTargets = [mTargets; [str2num(lineSplit{1}{2}) str2num(lineSplit{1}{3}) str2num(lineSplit{1}{4})]];
    line = fgets(fid);
end
fclose(fid);
save(['input_data_ArSenL_We_' num2str(ngram) '.mat']);
