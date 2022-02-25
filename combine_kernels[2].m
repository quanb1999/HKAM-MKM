function result = combine_kernels(weights, kernels)
    % length of weights should be equal to length of matrices
    % kernels_local,gamma
    m = length(weights);
    %n = size(kernels_local(1));
    result = zeros(size(kernels(:,:,1)));    
    
    for i=1:m
        result = result + weights(i) * kernels(:,:,i);
            
            
    end
end