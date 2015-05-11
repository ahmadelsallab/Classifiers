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
function [f, df] = CG_CLASSIFY_WE(VV, Dim, XX, target);

l = Dim';
N = size(XX,1);

% Do decomversion.
N_layers = size(Dim, 1) - 2; % remove input and target layers
offset = 0;
for layer = 1 : N_layers
    % NW_weights{1} = We;
    NW_weights{layer} = reshape(VV((offset+1) : (offset+(l(layer)+1)*l(layer+1))), l(layer)+1, l(layer+1));       
    offset = offset + (l(layer)+1)*l(layer+1);
end
w_class = reshape(VV(offset+1:offset+(l(N_layers+1)+1)*l(N_layers+2)), l(N_layers+1)+1, l(N_layers+2));

%XX = NM_lookupWe(idx, We); To be done inside NM_neuralNetActivation

BP_layerInputData = XX;
XX = [XX ones(N,1)];


[activationTemp BP_wprobs] = NM_neuralNetActivation(BP_layerInputData, NW_weights);        

targetout = exp(BP_wprobs{N_layers}*w_class);
targetout = targetout./repmat(sum(targetout,2),1,size(target,2));
f = -sum(sum( target(:,1:end).*log(targetout))) ;

IO = (targetout-target(:,1:end));
Ix_class=IO; 
dw_class =  (BP_wprobs{N_layers})'*Ix_class; 

layer = N_layers;
Ix_upper = Ix_class;
w_upper = w_class;
%baseUnit = 0; % at top level there's intermediate weight not base unit
while (layer >= 1)
	
	% delta_k = Ix{layer}
	% delta_j = delta_k * wJk' * f'(yink) = (Ix_upper*w_upper').*BP_wprobs{layer}.*(1-BP_wprobs{layer})
	%if (baseUnit == 0)


    global sActivationFunction;
    switch(sActivationFunction)
        case 'tanh'
            Ix{layer} = (Ix_upper*w_upper').*BP_wprobs{layer}.*(BP_wprobs{layer});
        case 'sigmoid'
            Ix{layer} = (Ix_upper*w_upper').*BP_wprobs{layer}.*(1-BP_wprobs{layer});
        case 'linear'
            % y_ink = w'*x;                    
            % If y_in = 0, f'(y_ink) = 0
            % Else f'(yink) = 1;
            % Get the zeros of the yink and multiply by them:
            % Note that: inside NM_layerActivation, activation is
            % set such that -ve values are dumped to 0's already
            % so what we need is to reconstruct matrix similar to BP_wprobs
            % but with 1's at the non-zeros to simulate f'(yink) = 1;
            f_dash_yink = BP_wprobs{layer};
            f_dash_yink(find(f_dash_yink > 0)) = 1;
            Ix{layer} = (Ix_upper*w_upper').*f_dash_yink;
    end            
    Ix{layer} = Ix{layer}(:,1:end-1);

    
	if(layer == 2)
        dw{layer} = XX'*Ix{layer};
    else if(layer == 1)
		% dWe = dw{1}
        dw{layer} = 0.*NW_weights{layer};
        % XX(i) = idx = [10, 150, 390, 40, 34] --> word indices
        % XX--> Ncases x N, N = n-grams = 5 for example
        for i = 1 : size(XX, 1)
            offset = 1;
            % N-gram = size(XX, 2) --> XX(i) = idx = [10, 150, 390, 40, 34] --> word indices
            % N * embedding_size = size(We, 2) = size(NW_weights{layer}, 2)
            embedding_size = size(NW_weights{layer}, 2) / size(XX, 2);
            for j = 1 : size(XX, 2) - 1
                %dWe(word_idx) += delta_i;
                % Update only the entry of We that corresponds to the indexed word XX(i,j), with the error of this example i = Ix{layer}(i, :)
                % size(dw{layer}(XX(i,j), :)) = size(Ix{layer}(i, :)) = embedding size
                dw{layer}(XX(i,j), :) = dw{layer}(XX(i,j), :) + Ix{layer}(offset:offset + embedding_size, :);  
                offset = offset + embedding_size + 1;
            end
        end
        
	else
		dw{layer} = (BP_wprobs{layer-1})'*Ix{layer};
    end
    Ix_upper = [];
	Ix_upper = Ix{layer};
	w_upper = [];
	w_upper = NW_weights{layer}; % in case of "base unit", NW_weights are the intermediate weights
    layer = layer - 1;
	% if(depthBaseUnitMapping ~= 0)
		% baseUnit = ~baseUnit; %switch to intermediate weight or base unit
	% end
	
end

df = [];
for(layer = 1 : N_layers)
	df = [df dw{layer}(:)'];
end
df = [df dw_class(:)'];
df = df';
