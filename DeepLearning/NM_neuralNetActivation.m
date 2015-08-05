% Function:
% Gives the activation of each network/classifier layer, both in augmented and raw form
% Inputs:
% weight: Matrix cell array {k}nxm, where k is the number of layers, n is
% the number of input neurons and m is the number of output neurons for
% each layer
% data: Matrix nxm, where n is the number of examples and m is the lenght of each example. data is not augmented
% Output:
% activation: Cell array of layers activations without ones colomn augmentation
% augmentedActivation: same as activation with ones colomn augmentation

function [activation augmentedActivation] = NM_neuralNetActivation(data, weights)
  rawLayerInputData = data;
  data = [data ones(size(data, 1) ,1)];
  layerInputData = data;

  for(layer = 1 : size(weights, 2))
	global bWordEmbedding;
    global nGram;
    global nWordEmbeddingSize;
    if(bWordEmbedding && layer == 1)
       layerActivation = zeros(size(rawLayerInputData, 1), nGram * nWordEmbeddingSize);
       for i = 1 : size(rawLayerInputData, 1)
           Xe = [];
           for j = 1 : nGram
              Xe = [Xe NM_lookupWe(rawLayerInputData(i, j), weights{layer})];
              
           end
            global sActivationFunction;
            switch(sActivationFunction)
                case 'tanh'                   
                    Xe = tanh(Xe);

                case 'sigmoid'
                    Xe = 1./(1 + exp(Xe));
                case 'linear'
                    Xe(find(Xe < 0)) = 0;
            end


            layerActivation(i, :) = Xe;
            augmentedLayerActivation = [layerActivation ones(size(rawLayerInputData,1), 1)];
       end
    else
        [layerActivation augmentedLayerActivation]= NM_layerActivation(layerInputData, weights{layer});
    end

    activation{layer} = layerActivation;
    augmentedActivation{layer} = augmentedLayerActivation;
    
    layerInputData = [];
    layerInputData = augmentedLayerActivation;
    
  end

end