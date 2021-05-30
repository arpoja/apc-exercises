function W = simpleNewton(w0, F, M, K)

% Newton-type iteration for zeroing given function F
% - Given a Hessian matrix or an approximation, M
% - Given a max number of iterations, K
% - Given an initial approximation, w0 

W = nan(numel(w0),K);
W(:,1) = w0;

for k = 1 : K
    W(:,k+1) = W(:,k) - 0.2*full( M(W(:,k)) \ F(W(:,k)) );
end 

end
