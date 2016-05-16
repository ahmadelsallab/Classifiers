% Function:
% Performs back propagation for the curent epoch on the all batches
% Inputs:
% CONFIG_strParams: Configuration parameters
% NM_strNetParams: Net parameters to update
% mTrainBatchData, mTrainBatchTargets: Training set. See BM_makeBatches
% nEpoch: The current iteration number
% vLayersSize: The sizes of net layers
% cPrevWeights: Weights of the previous learning phase (not previous epoch)
% mPrevClassWeights: Class weights of the previous learning phase (not previous epoch)
% nPhase, nNumPhases, bMapping: See CLS_fineTuneAndClassifyDNN
% sFeaturesFileName: String of the input txt file
% nBitfieldLength: The bitfield length of the Raw features
% vChunkLength: Vector storing the boundaries of each features chunk
% vOffset: The bitfield boundaries corresponding to each chunk
% eFeaturesMode: Raw, Normal, Binary, Bitfield
% Output:
% cWeights: Updated weights of each layer of unit
% NM_strNetParams: Updated Net parameters
function [NM_strNetParams] = BP_startBackProp_SGD(nBPNumExamplesInMiniBatch, NM_strNetParams, mTrainBatchData, mTrainBatchTargets,...
                                                  nEpoch, momentum, lrate)
        
        % Obtain training set sizes
        [nNumExamplesPerBatch nNumFeaturesPerExample nNumBatches] = size(mTrainBatchData);
        
        
        % Initialize variables
        nMiniBatchIndex = 0;
        
        % Initialize dw and dw_class
        for nLayer = 1 : size(NM_strNetParams.cWeights, 2)
            dw{nLayer} = 0 .* NM_strNetParams.cWeights{nLayer};
        end
        dw_class = 0 .* NM_strNetParams.mClassWeights;

        % Start the loop on the number of mini batches. The loop terminates
        % when all minibatches inside numbatches end. So loop end =
        % nNumBatches/nBPNumExamplesInMiniBatch
        for nBatchNum = 1 : nNumBatches / nBPNumExamplesInMiniBatch

            fprintf(1,'epoch %d batch %d\r',nEpoch,nBatchNum);

            %%%%%%%%%%% COMBINE nBPNumExamplesInMiniBatch MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Get the next minibatch index
            nMiniBatchIndex = nMiniBatchIndex + 1; 

            % Combine minibatches into one data/target matrices
            mCurrTrainMiniBatchData=[];
            mCurrTrainMiniBatchTargets=[]; 
            for ctrMiniBatch = 1 : nBPNumExamplesInMiniBatch

                % Augment Batch data
                mCurrTrainMiniBatchData = [mCurrTrainMiniBatchData 
                                           mTrainBatchData(:,:,(nMiniBatchIndex-1) * nBPNumExamplesInMiniBatch + ctrMiniBatch)]; 
                
                % Augment Batch targets
                mCurrTrainMiniBatchTargets = [mCurrTrainMiniBatchTargets
                                              mTrainBatchTargets(:,:,(nMiniBatchIndex-1)*nBPNumExamplesInMiniBatch + ctrMiniBatch)];
            end 
            


            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% START MIMIZER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % Start the minimizer                                                                        
            [dw, dw_class] = SGD_CLASSIFY(NM_strNetParams.cWeights, NM_strNetParams.mClassWeights, mCurrTrainMiniBatchData, mCurrTrainMiniBatchTargets,                              momentum, lrate, dw, dw_class);                                                       


            
            % Update weights
                                               
            % Net weights
            for(layer = 1 : NM_strNetParams.nNumLayers)
                NM_strNetParams.cWeights{layer} = NM_strNetParams.cWeights{layer} - dw{layer};
            end                               

            % Class weights
            NM_strNetParams.mClassWeights = NM_strNetParams.mClassWeights - dw_class;
                                    
							
            end
    %%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        end % end of minibatches loop
     
     
end % end function