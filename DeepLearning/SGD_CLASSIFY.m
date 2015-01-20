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
function [dw, dw_class] = SGD_CLASSIFY(NW_weights, w_class, XX, target, eMappingMode, NW_unitWeights, NW_weights_in, w_class_in, momentum, lrate, dw_old, dw_class_old);

N = size(XX, 1);
N_layers = size(NW_weights, 2);

switch(eMappingMode)
    case 'DEPTH_BASE_UNIT_MAPPING'
        N_layers = size(NW_weights_in, 2);
        NW_weights = NW_weights_in;
        w_class = w_class_in;

    otherwise

        
end % end-switch

BP_layerInputData = XX;
XX = [XX ones(N,1)];

switch(eMappingMode)
    case 'DEPTH_BASE_UNIT_MAPPING'
        [activationTemp BP_wprobs unitInternalActivations NW_unitWProbs] = NM_compositeNetActivation(BP_layerInputData, NW_unitWeights);
    otherwise
        [activationTemp BP_wprobs] = NM_neuralNetActivation(BP_layerInputData, NW_weights);        
end % end-switch

targetout = exp(BP_wprobs{N_layers}*w_class);
targetout = targetout./repmat(sum(targetout,2),1,size(target,2));
f = -sum(sum( target(:,1:end).*log(targetout))) ;

IO = (targetout-target(:,1:end));
Ix_class=IO;

dw_class =  momentum * dw_class_old + lrate * (BP_wprobs{N_layers})' * Ix_class; 
%dw_class(abs(dw_class) > 10) = 0.1*randn(size(dw_class(abs(dw_class) > 10), 1), size(dw_class(abs(dw_class) > 10), 2));
%dw_class(isnan(dw_class)) = 0.1*randn(size(dw_class(isnan(dw_class)), 1), size(dw_class(isnan(dw_class)), 2));
%dw_class(abs(dw_class) > 10) = 1;
%dw_class(isnan(dw_class)) = 1;

%dw_class(abs(dw_class) > 1) = ones(size(dw_class(abs(dw_class) > 1), 1), size(dw_class(abs(dw_class) > 1), 2));
%dw_class(isnan(dw_class)) = ones(size(dw_class(isnan(dw_class)), 1), size(dw_class(isnan(dw_class)), 2));

dw_class(abs(dw_class) > 10) = 1000*randn(size(dw_class(abs(dw_class) > 10), 1), size(dw_class(abs(dw_class) > 10), 2));
dw_class(isnan(dw_class)) = 1000*randn(size(dw_class(isnan(dw_class)), 1), size(dw_class(isnan(dw_class)), 2));

layer = N_layers;
Ix_upper = Ix_class;
w_upper = w_class;
%baseUnit = 0; % at top level there's intermediate weight not base unit
while (layer >= 1)
	
	% delta_k = Ix{layer}
	% delta_j = delta_k * wJk' * f'(yink) = (Ix_upper*w_upper').*BP_wprobs{layer}.*(1-BP_wprobs{layer})
	%if (baseUnit == 0)

    switch(eMappingMode)
        case 'DEPTH_BASE_UNIT_MAPPING'
            Ix{layer} = BP_deltaUnit(NW_unitWeights{layer}, NW_unitWProbs{layer}, Ix_upper, w_upper);
            
        otherwise
            global sActivationFunction;
            switch(sActivationFunction)
                case 'tanh'
                    Ix{layer} = (Ix_upper*w_upper').*BP_wprobs{layer}.*(BP_wprobs{layer});
                case 'sigmoid'
                     Ix{layer} = (Ix_upper*w_upper').*BP_wprobs{layer}.*(1-BP_wprobs{layer});
            end            
            Ix{layer} = Ix{layer}(:,1:end-1);

    end % end-switch
    
	if(layer ~= 1)
		%dw{layer} = (BP_wprobs{layer-1})'*Ix{layer};
        dw{layer} = momentum * dw_old{layer} + lrate * (BP_wprobs{layer-1})' * Ix{layer};
	else
        %dw{layer} = XX'*Ix{layer};
        dw{layer} = momentum * dw_old{layer} + lrate * XX'*Ix{layer};
		
    end
    %dw{layer}(abs(dw{layer}) > 10) = 0.1*randn(size(dw{layer}(abs(dw{layer}) > 10), 1), size(dw{layer}(abs(dw{layer}) > 10), 2));
    %dw{layer}(isnan(dw{layer})) = 0.1*randn(size(dw{layer}(isnan(dw{layer})), 1), size(dw{layer}(isnan(dw{layer})), 2));
    
    %dw{layer}(abs(dw{layer}) > 10) = 1;    
    %dw{layer}(isnan(dw{layer})) = 1;

    %dw{layer}(abs(dw{layer}) > 1) = ones(size(dw{layer}(abs(dw{layer}) > 1), 1), size(dw{layer}(abs(dw{layer}) > 1), 2));
    %dw{layer}(isnan(dw{layer})) = ones(size(dw{layer}(isnan(dw{layer})), 1), size(dw{layer}(isnan(dw{layer})), 2));

    dw{layer}(abs(dw{layer}) > 10) = 1000*randn(size(dw{layer}(abs(dw{layer}) > 10), 1), size(dw{layer}(abs(dw{layer}) > 10), 2));
    dw{layer}(isnan(dw{layer})) = 1000*randn(size(dw{layer}(isnan(dw{layer})), 1), size(dw{layer}(isnan(dw{layer})), 2));

    
    Ix_upper = [];
	Ix_upper = Ix{layer};
	w_upper = [];
	w_upper = NW_weights{layer}; % in case of "base unit", NW_weights are the intermediate weights
    layer = layer - 1;
	% if(depthBaseUnitMapping ~= 0)
		% baseUnit = ~baseUnit; %switch to intermediate weight or base unit
	% end
	
end


