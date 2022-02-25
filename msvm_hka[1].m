function [predict_y,Scores,kernel_weights] = msvm_hka(train_x,feature_id,train_y,test_x,test_y,c,gamma_list,k,lamda,IsMK)
predict_y=[];
Scores=[];
kernel_weights=[];
m = size(feature_id,1);
num_train_samples = size(train_x,1);
num_test_samples = size(test_x,1);

%1.computer global training kernels (with RBF)
K_train=[];
for i=1:m

	kk_train = kernel_RBF(train_x(:,feature_id(i,1):feature_id(i,2)),train_x(:,feature_id(i,1):feature_id(i,2)),gamma_list(i));
	K_train(:,:,i)=kk_train;	
end


if strcmp(IsMK,'MKL-HKA')
	[kernel_weights] = compute_kernels_weights(K_train,train_y,k,lamda);
elseif strcmp(IsMK,'MKL-MW')
	kernel_weights = ones(m,1);kernel_weights = kernel_weights/m;
elseif strcmp(IsMK,'K-GE')
	kernel_weights = zeros(m,1);kernel_weights(1)=1;
elseif strcmp(IsMK,'K-MCD')
	kernel_weights = zeros(m,1);kernel_weights(2)=1;
elseif strcmp(IsMK,'K-NMBAC')
	kernel_weights = zeros(m,1);kernel_weights(3)=1;
elseif strcmp(IsMK,'K-PSSM-AB')
	kernel_weights = zeros(m,1);kernel_weights(4)=1;
elseif strcmp(IsMK,'K-PSSM-Pse')
	kernel_weights = zeros(m,1);kernel_weights(5)=1;
elseif strcmp(IsMK,'K-PSSM-DWT')
	kernel_weights = zeros(m,1);kernel_weights(6)=1;
end

all_weight = sum(kernel_weights);
for i = 1:m
   kernel_weights(i) = kernel_weights(i)/all_weight;
end

K_train_com = combine_kernels(kernel_weights, K_train);

%4.computer global testing kernels (with RBF)
K_test=[];
for i=1:m

	kk_test = kernel_RBF(test_x(:,feature_id(i,1):feature_id(i,2)),train_x(:,feature_id(i,1):feature_id(i,2)),gamma_list(i));
	K_test(:,:,i)=kk_test;	
end
K_test_com = combine_kernels(kernel_weights, K_test);


Y = train_y';

options = optimset;      
options.LargeScale = 'on'; 
options.Display = 'off'; 
  
 
n = length(Y);    
H = (Y'*Y).*K_train_com; 
H=(H+H')/2;     
f = -ones(n,1);  
A = [];  
b = [];  
Aeq = Y;  
beq = 0;  
lb = zeros(n,1);  
mu_c = c; 
ub = mu_c.*ones(n,1); 
a0 = zeros(n,1);    
[a,fval,eXitflag,output,lambda]  = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);  

  
epsilon = 1e-3;    
  
sv_label = find(abs(a)>epsilon);       
svm.a = a(sv_label);  
%svm.Xsv = X(:,sv_label);  
svm.Ysv = Y(sv_label);  
svm.svnum = length(sv_label); 
svm.sv_indices = sv_label;

%Xt = test_x'; Yt = test_y';
temp = (svm.a'.*svm.Ysv)*K_train_com(sv_label,sv_label);  
%total_b = svm.Ysv-temp;  
b = mean(svm.Ysv-temp); 

w = (svm.a'.*svm.Ysv)*K_test_com(:,sv_label)';
result_score = (w + b)';  

predict_y = sign(w+b);  %f(x)  
predict_y = predict_y';  
Scores =  1./(1 + exp(-1*result_score));

end

%RBF kernel function
function k = kernel_RBF(X,Y,gamma)
	r2 = repmat( sum(X.^2,2), 1, size(Y,1) ) ...
	+ repmat( sum(Y.^2,2), 1, size(X,1) )' ...
	- 2*X*Y' ;
	k = exp(-r2*gamma); 
end


 