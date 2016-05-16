% Function:
% Makes error back propagation for the NN
% Inputs:
% VV: Vector of all class weights serialized row-wise
% Dim: Vector of sizes of top layer (input and output)
% XX: Vector of the input data to the NN taken row-wise
% wTopProbs: The activations at the input of the top layer
% target: The associated target
% eMappingMode: See CONFIG_setConfigParams
% NW_unitWeights: Cell array of weights of each constituting unit of the net
% NW_weights_in: Weights of each layer
% w_class_in: Weights of the top class layer
% Output:
% f: The negative of the error
% df: The back-propagated delta (to be multiplied by input data to update
% the weigths
function [dw, dw_class] = SGD_CLASSIFY(weights, w_class, XX, target, momentum, lrate, dw_old, dw_class_old);

N_layers = length(weights) + 1;

BP_layerInputData = XX;
% For bias
XX = [XX ones(N,1)];

% Feed forward in the network
[activationTemp BP_wprobs] = NM_neuralNetActivation(BP_layerInputData, NW_weights);
% Get the top layer output
targetout = exp(BP_wprobs{N_layers}*w_class);
% Normalize the output
targetout = targetout./repmat(sum(targetout,2),1,size(target,2));

% Error at upper layer
IO = (targetout-target(:,1:end));
% Contribution at each neuron is the same as its error
Ix_class=IO;
% The delta of th upper layer weights is just function of the upper layer errors. This shall propagate
dw_class =  momentum * dw_class_old + lrate * (BP_wprobs{N_layers})' * Ix_class; 


% The best practice is to restart the weights with randoms if they increase above certain limit, or Inf
dw_class(abs(dw_class) > 10) = 1000*randn(size(dw_class(abs(dw_class) > 10), 1), size(dw_class(abs(dw_class) > 10), 2));
dw_class(isnan(dw_class)) = 1000*randn(size(dw_class(isnan(dw_class)), 1), size(dw_class(isnan(dw_class)), 2));

% Start with upper layer error
layer = N_layers;
Ix_upper = Ix_class;
w_upper = w_class;

% BAck propagate
while (layer >= 1)
	
	% delta_k = Ix{layer}
	% delta_j = delta_k * wJk' * f'(yink) = (Ix_upper*w_upper').*BP_wprobs{layer}.*(1-BP_wprobs{layer})
	
    global sActivationFunction;
    switch(sActivationFunction)
        case 'tanh'
            Ix{layer} = (Ix_upper*w_upper').*BP_wprobs{layer}.*(BP_wprobs{layer});
        case 'sigmoid'
             Ix{layer} = (Ix_upper*w_upper').*BP_wprobs{layer}.*(1-BP_wprobs{layer});
    end            
    Ix{layer} = Ix{layer}(:,1:end-1);


    % Apply momentum and learning rate
	if(layer ~= 1)
		%dw{layer} = (BP_wprobs{layer-1})'*Ix{layer};
        dw{layer} = momentum * dw_old{layer} + lrate * (BP_wprobs{layer-1})' * Ix{layer};
	else
        %dw{layer} = XX'*Ix{layer};
        dw{layer} = momentum * dw_old{layer} + lrate * XX'*Ix{layer};
		
    end
    
    % The best practice is to restart the weights with randoms if they increase above certain limit, or Inf
    dw{layer}(abs(dw{layer}) > 10) = 1000*randn(size(dw{layer}(abs(dw{layer}) > 10), 1), size(dw{layer}(abs(dw{layer}) > 10), 2));
    dw{layer}(isnan(dw{layer})) = 1000*randn(size(dw{layer}(isnan(dw{layer})), 1), size(dw{layer}(isnan(dw{layer})), 2));

    % Take the current layer as the upper layer to the next lower one
    Ix_upper = [];
	Ix_upper = Ix{layer};
	w_upper = [];
	w_upper = weights{layer}; % in case of "base unit", weights are the intermediate weights
    layer = layer - 1;

	
end


