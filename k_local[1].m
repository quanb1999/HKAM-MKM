 function [Klocal_C,Klocal_y] = k_local(K_C,K_y,k)
   k_near = k_nearest(K_C,k);
   for v = 1:size(K_y,1)
   for i =1:k
       for j=1:k
          Klocal_C(v,i,j,:)=K_C(k_near(v,i),k_near(v,j),:);
          Klocal_y(v,i,j)=K_y(k_near(v,i),k_near(v,j));
       end
   end
   end
 end
 
 function k_neighbors = k_nearest(X,k)
n = size(X,1);

dis = zeros(n,n);
sort_dis = zeros(n,n);
ix = zeros(n,n);
k_neighbors = zeros(n,k);

%compute distance
 for i = 1:n
    for j =i:n
       dis(i,j)=sqrt(sum((X(i,:)-X(j,:)).^2));
       dis(j,i) = dis(i,j);
    end
 end 

%sort and select k neighbors
for i = 1:n
    [sort_dis(i,:),ix(i,:)]=sort(dis(i,:),'ascend');
    for j=2:k+1
      k_neighbors(i,j-1) = ix(i,j);
    end
end

end
