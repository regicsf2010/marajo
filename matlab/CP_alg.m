function [ys,W] = CP_alg(mixtures)

n = 10; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPUTE V AND U.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set short and long half-lives.
shf 		= 1; 
lhf 		= 900000; 	

% Define max mask len for convolution.
max_mask_len= 50;

% Short-term mask.
h=shf; t = n*h; lambda = 2^(-1/h); temp = [0:t-1]'; 
mask = lambda.^temp;
mask(1) = 0; mask = mask/sum(abs(mask));  mask(1) = -1;
s_mask=mask;

% Long-term mask.
h=lhf;t = n*h; t = min(t,max_mask_len); t=max(t,1);
lambda = 2^(-1/h); temp = (0:t-1)';
mask = lambda.^temp;
mask(1) = 0; mask = mask/sum(abs(mask));  mask(1) = -1;
l_mask=mask;

% Filter each column of mixtures array.
S=filter(s_mask,1,mixtures); 	
L=filter(l_mask,1,mixtures);

% Find short-term and long-term covariance matrices.
U=cov(S,1);		
V=cov(L,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NOW USE W MATRIX FROM EIG FUNCTION TO EXTRACT **ALL** SOURCES.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find optimal solution as eigenvectors W.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[W d]=eig(V,U); W=real(W);

ys = -(mixtures*W);

end