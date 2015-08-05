clear, clc, close all;


% Train We 
cd('C:\Users\ASALLAB\Google Drive\PostDoc\Code\sentimentanalysis\classifiers\Configurations\');
run_We;

% Train ArSenL embedding
cd('C:\Users\ASALLAB\Google Drive\PostDoc\Code\sentimentanalysis\classifiers\Configurations\');
run_ArSenL_Embedding;

% Copy: final_net_ArSenL_embedding.mat, input_data_ArSenL_Embedding_1.mat, vocab_ArSenL_Embedding into RAE/data/ATB_ArSenL_We
% Run run_ArSenL_We in RAE/code
