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
  data = [data ones(size(data, 1) ,1)];
  layerInputData = data;

  for(layer = 1 : size(weights, 2))
	global bWordEmbedding;
    if(bWordEmbedding && layer == 1)
       layerActivation = zeros(size(layerInputData, 1), size(layerInputData, 2) * size(weights{layer}, 2));
       for i = 1 : size(layerInputData, 1)
           Xe = zeros(1, size(layerInputData, 2) * size(weights{layer}, 2));;
           for j = 1 : size(layerInputData, 2)
              Xe = [Xe NM_lookupWe(layerInputData(i, j), weights{layer})]
              
           end
           layerActivation(i, :) = Xe;
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