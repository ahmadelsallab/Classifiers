% Function:
% - Makes initialization of the top class layer
% - Runs backpropagation
% Inputs:
% NM_strNetParams: The net parameters to be tuned
% CONFIG_strParams: The configurations
% TST_strPerformanceInfo: The structure to update test results in it
% mTrainBatchTargets, mTrainBatchData, mTestBatchTargets, mTestBatchData: see BM_makeBatches
% nPhase: The current phase of classifier re-use (mapping)
% nNumPhases: Total number of phases for classifier re-use (mapping)
% hFidLog: Handle of the log file
% bMapping: Is classifier re-use enabled
% nBitfieldLength: The bitfield length of the Raw features
% vChunkLength: Vector storing the boundaries of each features chunk
% vOffset: The bitfield boundaries corresponding to each chunk
% eFeaturesMode: Raw, Normal, Binary, Bitfield
% Output:
% None
function [NM_strNetParams, TST_strPerformanceInfo] = CLS_fineTuneAndClassifyDNN(NM_strNetParams, CONFIG_strParams, TST_strPerformanceInfo,...
                                                                                mDevTargets, mDevFeatures,...
                                                                                mTrainBatchTargets, mTrainBatchData, mTestBatchTargets, mTestBatchData,...
                                                                                nPhase, nNumPhases, hFidLog, bMapping,...
                                                                                nBitfieldLength, vChunkLength, vOffset, eFeaturesMode)

    % The Backprop epochs is different according to mapping phase
    if bMapping == 0
        CLS_strPrvt.nBPMaxEpoch = CONFIG_strParams.nBPNumEpochsBeforeMapping;
    else
        CLS_strPrvt.nBPMaxEpoch = CONFIG_strParams.nBPNumEpochsDuringMapping;
    end % end if-else

    % Initialize test error to huge number
    TST_strPerformanceInfo.nMinTestError = 1000000;
    TST_strPerformanceInfo.nMinTrainError = 1000000;
    
    fprintf(1,'\nTraining discriminative model on the train dataset by minimizing cross entropy error. \n');
    %load(CONFIG_strParams.sInputDataWorkspace);
    % The number of targets is the 2nd dimension of the 3-D matrix mTrainBatchTargets
    CLS_strPrvt.nNumTargets = size(mTrainBatchTargets, 2);
        
    %%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
    % If mapping phases have started then the class weights are initialized
    % to their previous, otherwise null
    if(bMapping == 1)
        NM_strNetParams.mClassWeights = NM_strNetParams.mClassWeights;
    else
        NM_strNetParams.mClassWeights = []; % shall be initialized to random inside NM_initializeClassLayer
    end
    
    % Initialize the class layer weigths
    [NM_strNetParams.mClassWeights, CLS_strPrvt.mPrevClassWeights] =...
        NM_initializeClassLayer(CONFIG_strParams.eMappingMode, CONFIG_strParams.eMappingDirection, bMapping,...
                                CONFIG_strParams.sSVMWorkSpaceFileName, NM_strNetParams.mClassWeights, nPhase,...
                                nNumPhases, NM_strNetParams.cWeights{NM_strNetParams.nNumLayers}, CLS_strPrvt.nNumTargets);
    % Keep the previous weights before training. This is valid only in case
    % of mapping has started
    if bMapping == 1
        if (strcmp(CONFIG_strParams.eMappingDirection, 'DEPTH') && strcmp(CONFIG_strParams.eMappingMode, 'DEPTH_CASCADED_DATA_REPRESENTATION_MAPPING'))
            CLS_strPrvt.cPrevWeights = NM_strNetParams.cCascadedBaseUnitWeights;
        else
            CLS_strPrvt.cPrevWeights = NM_strNetParams.cWeights;
        end
    else
        CLS_strPrvt.cPrevWeights = [];
    end

    for(nLayer = 1 : NM_strNetParams.nNumLayers) % nNumLayers is execluding the input data and target layers
        CLS_strPrvt.vLayersSize(nLayer) = size(NM_strNetParams.cWeights{nLayer}, 1) - 1; % we have to remove the bias (-1)
    end
    % nNumLayers is execluding the input data and target layers, so
    % remaining 2 layers to initialize
    CLS_strPrvt.vLayersSize(NM_strNetParams.nNumLayers + 1) = size(NM_strNetParams.mClassWeights, 1) - 1; % we have to remove the bias (-1)
    CLS_strPrvt.vLayersSize(NM_strNetParams.nNumLayers + 2) = CLS_strPrvt.nNumTargets;

    TST_strPerformanceInfo.vTestErr=[];
    TST_strPerformanceInfo.vTrainErr=[];

    %%%%%%%%%% END OF PREINITIALIZATION OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%% END OF PREINITIALIZATION OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Set previous dev err to max value
    nPrevDevErr = 100000;

    % Initialize the learning rate
    lrate = 0.1;
    minrate = 0.000001;

    %%%%%%%%%%%%%%%%%%%%%%%% START FINE TUNING  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for nEpoch = 1 : CLS_strPrvt.nBPMaxEpoch
        if(nEpoch < 5)
            momentum = 0.0;
        else
            momentum = 0.9;
        end;
        % If bBPKeepMinWeightsEveryEpoch is configured 1 then set the
        % previous weights (Net and Class) to the minimum error weights reached ever
        if ((bMapping == 1) && (CONFIG_strParams.bBPKeepMinWeightsEveryEpoch == 1))
            for(nLayer = 1 : NM_strNetParams.nNumLayers)
                CLS_strPrvt.cPrevWeights{nLayer} = CLS_strPrvt.cMinErrWeights{nLayer};
            end
            CLS_strPrvt.mPrevClassWeights = CLS_strPrvt.mClassWeightsMinErr;
        end
    %%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [TST_strPerformanceInfo.vTrainErr(nEpoch)] =...
            TST_computeClassificationErrDNN(mTrainBatchData, mTrainBatchTargets, NM_strNetParams, bMapping, CONFIG_strParams.eMappingDirection,...
                                            CONFIG_strParams.eMappingMode, nPhase, nNumPhases, 'EPOCH_ERR_CALC', nBitfieldLength, vChunkLength, vOffset, eFeaturesMode);
    %%%%%%%%%%%%%% END OF COMPUTING TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [TST_strPerformanceInfo.vTestErr(nEpoch), vObtainedTestTargets, vDesiredTestTargets] =...
            TST_computeClassificationErrDNN(mTestBatchData, mTestBatchTargets, NM_strNetParams, bMapping, CONFIG_strParams.eMappingDirection,...
                                         CONFIG_strParams.eMappingMode, nPhase, nNumPhases, 'EPOCH_ERR_CALC', nBitfieldLength, vChunkLength, vOffset, eFeaturesMode);
                                         
        % Build confusion matrix
        [TST_strPerformanceInfo.mTestConfusionMatrix, TST_strPerformanceInfo.mTestNormalConfusionMatrix, TST_strPerformanceInfo.vTestNumTrainExamplesPerClass,...
        TST_strPerformanceInfo.vTestAccuracyPerClass, TST_strPerformanceInfo.nTestOverallAccuracy] = LM_buildConfusionMatrix(vDesiredTestTargets, vObtainedTestTargets);
        
        % Calculate precision
        pr = TST_strPerformanceInfo.mTestConfusionMatrix(1,1)/(TST_strPerformanceInfo.mTestConfusionMatrix(1,1) + TST_strPerformanceInfo.mTestConfusionMatrix(2,1));

        % Calculate recall
        re = TST_strPerformanceInfo.mTestConfusionMatrix(1,1)/(TST_strPerformanceInfo.mTestConfusionMatrix(1,1) + TST_strPerformanceInfo.mTestConfusionMatrix(1,2));       
        
        % Calculate F1 score
        F1 = 2 * (pr*re)/(pr+re);
        
        fprintf(1, 'Recall = %d, Precision = %d, F1 score = %d\n', re, pr, F1);
        fprintf(hFidLog, 'Recall = %d, Precision = %d, F1 score = %d\n', re, pr, F1);
                                         
    %%%%%%%%%%%%%%% END OF COMPUTING TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [nNumTrainExamples nNumTrainFeatures nNumTrainBatches]=size(mTrainBatchData);
        [nNumTestExamples nNumTestFeatures nNumTestBatches]=size(mTestBatchData);

        if(strcmp(eFeaturesMode, 'Raw'))
            nNumTrainFeatures = nBitfieldLength;
            nNumTestFeatures = nBitfieldLength;
        end

        % Log the result to display and log file
        fprintf(hFidLog,'Before epoch %d Train # misclassified: %d (from %d). Test # misclassified: %d (from %d) \t \t \n',...
                nEpoch, TST_strPerformanceInfo.vTrainErr(nEpoch), nNumTrainExamples * nNumTrainBatches, TST_strPerformanceInfo.vTestErr(nEpoch),nNumTestExamples * nNumTestBatches);

        fprintf(1,'Before epoch %d Train # misclassified: %d (from %d). Test # misclassified: %d (from %d) \t \t \n',...
                nEpoch, TST_strPerformanceInfo.vTrainErr(nEpoch), nNumTrainExamples * nNumTrainBatches, TST_strPerformanceInfo.vTestErr(nEpoch),nNumTestExamples * nNumTestBatches);

        % Keep the weigths if error is minimized
        if TST_strPerformanceInfo.vTrainErr(nEpoch) <= TST_strPerformanceInfo.nMinTrainError
            TST_strPerformanceInfo.nMinTrainError = TST_strPerformanceInfo.vTrainErr(nEpoch);
            for(ctrLayer = 1 : NM_strNetParams.nNumLayers)
                CLS_strPrvt.cMinErrWeights{ctrLayer} = NM_strNetParams.cWeights{ctrLayer};
            end
            CLS_strPrvt.mClassWeightsMinErr = NM_strNetParams.mClassWeights;
        end
    %%%%%%%%%%%%%%%%%%%% START BACKPROP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
        switch(CONFIG_strParams.sMinimzerType)
            case 'CG'
                [cWeights, NM_strNetParams] = BP_startBackProp(CONFIG_strParams, NM_strNetParams, mTrainBatchData, mTrainBatchTargets,...
                                                               nEpoch, CLS_strPrvt.vLayersSize, CLS_strPrvt.cPrevWeights, CLS_strPrvt.mPrevClassWeights, nPhase, nNumPhases, bMapping,...
                                                               nBitfieldLength, vChunkLength, vOffset, eFeaturesMode);
            case 'SGD'
                
                % Keep old weights to roll back if needed
                CLS_strPrvt.cPrevWeights = NM_strNetParams.cWeights;
                CLS_strPrvt.mPrevClassWeights = NM_strNetParams.mClassWeights;
                
                [NM_strNetParams] = BP_startBackProp_SGD(CONFIG_strParams, NM_strNetParams, ...
                                                                   mTrainBatchData, mTrainBatchTargets,...
                                                                   nEpoch, CLS_strPrvt.vLayersSize, CLS_strPrvt.cPrevWeights, CLS_strPrvt.mPrevClassWeights, nPhase, nNumPhases, bMapping,...
                                                                   nBitfieldLength, vChunkLength, vOffset, eFeaturesMode, momentum, lrate);
                
                %%%%%%%%%%%%%%%%%%%% COMPUTE Development set MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                [nDevErr] =...
                    TST_computeClassificationErrDNN(mDevFeatures, mDevTargets, NM_strNetParams, bMapping, CONFIG_strParams.eMappingDirection,...
                                                    CONFIG_strParams.eMappingMode, nPhase, nNumPhases, 'EPOCH_ERR_CALC', nBitfieldLength, vChunkLength, vOffset, eFeaturesMode);

                % Calculate accuracy ratio
                nErrRatio = (nDevErr - nPrevDevErr)*1.0/(nPrevDevErr+1);
                % Check if dev err reduced
                if(nErrRatio <= 0.0)
                    % Accept this epoch
                    nPrevDevErr = nDevErr;
                    %lrate = lrate * 2.0;
                    fprintf(1, 'Epoch accepted, dev error ratio= %d, lrate = %d\n', nErrRatio, lrate);
                else
                    % Reduce learning rate
                    lrate = lrate / 2.0;
                    %lrate = lrate / 100.0;
                    % Roll back to old weights
                    NM_strNetParams.cWeights = CLS_strPrvt.cPrevWeights;
                    NM_strNetParams.mClassWeights = CLS_strPrvt.mPrevClassWeights;
                    % Check if min rate reached
                    if(lrate <= minrate)
                        fprintf(1, 'Minimum learning rate reached, dev error ratio= %d\n', nErrRatio);
                        % Stop
                        break;
                    else
                        fprintf(1, 'Epoch rejected, dev error ratio= %d, lrate = %d\n', nErrRatio, lrate);
                    end
                end


        end
    %%%%%%%%%%%%%%%%%%%% END BACKPROP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

         % If bBPKeepMinWeightsEveryEpoch is configured, then save the
         % weigths giving min. error each epoch
         if bMapping == 1 && CONFIG_strParams.bBPKeepMinWeightsEveryEpoch == 1
            for ctrLayer = 1 : NM_strNetParams.nNumLayers
                NM_strNetParams.cWeights{ctrLayer} = CLS_strPrvt.cMinErrWeights{ctrLayer};
            end
            NM_strNetParams.mClassWeights = CLS_strPrvt.mClassWeightsMinErr;
         end
    end % end of epoches loop
    
    % Keep weights giving min. error over all epochs
    for ctrLayer = 1 : NM_strNetParams.nNumLayers
        NM_strNetParams.cWeights{ctrLayer} = CLS_strPrvt.cMinErrWeights{ctrLayer};
    end
    NM_strNetParams.mClassWeights = CLS_strPrvt.mClassWeightsMinErr;

end % end function