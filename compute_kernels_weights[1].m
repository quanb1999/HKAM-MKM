function [w] = compute_kernels_weights(Kernels_list,adjmat,k,lamda)

%Kernels_list n¡Án¡Ám
%local_Kernels_list n¡Ák¡Ák¡Ám
%adjmat:n¡Á1   K_y :n¡Ák
% k : the number of neighbors

num_samples = size(Kernels_list,1);
num_kernels = size(Kernels_list,3);

y = adjmat;
ga = y*y';

[Kglo_C,Kglo_y] = global_kernel(Kernels_list,ga);
[Klocal_C,Klocal_y] = k_local(Kglo_C,Kglo_y,k);
N_U = size(ga,1);
l=ones(N_U,1);
H = eye(N_U) - (l*l')/N_U;

%compute M m¡Ám
M = zeros(num_kernels,num_kernels);
for i=1:num_kernels
	for j=1:num_kernels
		kk1 = H* Kglo_C(:,:,i)*H;
		kk2 = H* Kglo_C(:,:,j)*H;
        mm = trace(kk1'*kk2);
		m1 = trace(kk1*kk1');
		m2 = trace(kk2*kk2');
		M(i,j) = mm/(sqrt(m1)*sqrt(m2));
    end
end


%compute M_local n¡Ám¡Ám
M_local = zeros(num_samples,num_kernels,num_kernels);
for v=1:num_samples 
    for i=1:num_kernels
        for j=1:num_kernels
            kk1 = squeeze(Klocal_C(v,:,:,i));
            kk2 = squeeze(Klocal_C(v,:,:,j));
            mm = trace(kk1'*kk2);
            M_local(v,i,j) =mm;
        end
    end    
end

%Initialise ¦Ó,gamma
tau =ones(num_samples,1);
gamma = ones(num_kernels,1);

%compute Q,Z
Q = compute_Q(Klocal_C,Klocal_y,tau,num_kernels,num_samples);
Z = compute_Z(Kglo_C,Kglo_y,num_kernels,N_U,H);



[w,tau,obj,Q,Z] = iter_obj(M,M_local,Q,Z,gamma,lamda,Kglo_C,Kglo_y,Klocal_C,Klocal_y,tau,num_kernels,num_samples);
obj_1 =obj;

  while 1
    [w,tau,obj,Q,Z] = iter_obj(M,M_local,Q,Z,gamma,lamda,Kglo_C,Kglo_y,Klocal_C,Klocal_y,tau,num_kernels,num_samples);
     if((obj_1-obj)/obj<=1e-4)
         break;
     end
     obj_1 =obj;
  end 
end


% build global center kernel matrix K_C
 function [K_C,K_y] = global_kernel(K,K_Y)
  K_y = K_Y;
  for v = 1:size(K,3)
    n=size(K,1);
    f_ij = sum(diag(K(:,:,v)))/n^2;
    f_j = sum(K(:,:,v))/n;
    f_i = sum(K(:,:,v),2)/n;

    for i =1:n
       for j=i:n
          K_C(i,j,v)=K(i,j,v)-f_j(j)-f_i(i)+f_ij;
          K_C(j,i,v)=K_C(i,j,v);
       end
    end
  end
 end
 

function [J] = obj_function(M,Q,Z,gamma,lamda)
    J =gamma'*M*gamma -2*(1-lamda)*gamma'*Q -2*lamda*gamma'*Z;
end

%Update iteratively until convergence
function [w,tau,obj,Q,Z] = iter_obj(M,M_local,Q,Z,gamma,lamda,Kernels_list,ga,local_Kernels_list,K_y,tau,num_kernels,num_samples)

 %Solve QP problem and update gamma weight_v
 falpha = @(gamma)obj_function(M,Q,Z,gamma,lamda);        
 [gamma, fval] = optimize_weights(gamma, falpha);
 weight_v = gamma/norm(gamma,2);
 w = weight_v;

%compute objective value(hybrid kernel alignment value)
obj = ((1-lamda)*weight_v'*Q - lamda*weight_v'*Z)/(weight_v'*M*weight_v)^0.5;

%updata ¦Ó
tau = compute_tau(M,M_local,weight_v,num_samples);

%updata Q
Q = compute_Q(local_Kernels_list,K_y,tau,num_kernels,num_samples); 

end

function Q = compute_Q(local_Kernels_list,K_y,tau,num_kernels,num_samples)
Q = zeros(num_kernels,1);
  for i =1:num_kernels
     temp =0;
     for j =1:num_samples 
         temp = temp + trace(squeeze(local_Kernels_list(j,:,:,i))'*squeeze(K_y(j,:,:)))/tau(j);
     end
     Q(i) = temp/num_samples;
  end
end

function Z = compute_Z(Kernels_list,ga,num_kernels,N_U,H)
Z = zeros(num_kernels,1);
  for i =1:num_kernels
      kk = H*Kernels_list(:,:,i)*H;
     bb = trace(kk'*ga);
     Z(i) = size(Kernels_list,1)*bb*(N_U-1)^-2;
  end
end

function tau = compute_tau(M,M_local,weight_v,num_samples)
tau = zeros(num_samples,1);
  for i =1:num_samples 
     tau(i) =(weight_v'*squeeze(M_local(i,:,:))*weight_v)^0.5/(weight_v'*M*weight_v)^0.5;
  end

end



function [x, fval] = optimize_weights(x0, fun)
    n = length(x0);
    Aineq   = [];
    bineq   = [];
    Aeq     = [];
    beq     = [];
    LB      = zeros(n,1);
    UB      = [];
    options = optimoptions('fmincon','Algorithm','interior-point', 'Display', 'notify');
    [x,fval] = fmincon(fun,x0,Aineq,bineq,Aeq,beq,LB,UB,[],options);
end


