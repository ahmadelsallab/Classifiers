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

        % Keep old weights to roll back if needed
        CLS_strPrvt.cPrevWeights = NM_strNetParams.cWeights;
        CLS_strPrvt.mPrevClassWeights = NM_strNetParams.mClassWeights;
        
        [NM_strNetParams] = BP_startBackProp_SGD(CONFIG_strParams, NM_strNetParams, ...
                                                           mTrainBatchData, mTrainBatchTargets,...
                                                           nEpoch, CLS_strPrvt.vLayersSize, CLS_strPrvt.cPrevWeights, CLS_strPrvt.mPrevClassWeights, nPhase, nNumPhases, bMapping,...
                                                           nBitfieldLength, vChunkLength, vOffset, eFeaturesMode, momentum, lrate);
        
        %%%%%%%%%%%%%%%%%%%% COMPUTE Development set MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [nDevErr] =...
            TST_computeClassificationErrDNN(mDevFeatures, mDevTargets, NM_strNetParams);

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

 
    end % end of epoches loop
    


end % end function