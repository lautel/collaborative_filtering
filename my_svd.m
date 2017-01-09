function [Pe, Qe] = my_svd(riu, cluster, N, M, K, alpha, beta)

Pe(1:size(cluster,2)) = {rand(N,K)};    % items x K
Qe(1:size(cluster,2)) = {rand(K,M)};    % K x usuarios  

e=0;
for cl=1:size(cluster,2)
fprintf('Iteración en cluster %d \n',cl);
    Riu = riu(:,cluster{cl});
    N_ = size(Riu,1);
    M_ = size(Riu,2);
    
    for step=1:100      % iterations time
        for i=1:N_      
            for j=1:M_  
                
                if ~isnan(Riu(i,j))
                    eij = Riu(i,j) - Pe{1,cl}(i,:)*Qe{1,cl}(:,j);
                    for k=1:K
                        Pe{1,cl}(i,k) = Pe{1,cl}(i,k) + alpha * (2 * eij * Qe{1,cl}(k,j) - beta * Pe{1,cl}(i,k));
                        Qe{1,cl}(k,j) = Qe{1,cl}(k,j) + alpha * (2 * eij * Pe{1,cl}(i,k) - beta * Qe{1,cl}(k,j));
                    end
                end
                
            end
        end
                
        
        % Error updating

        for i=1:N_ 
            for j=1:M_ 
                
                if ~isnan(Riu(i,j))
                    for k=1:K
                        e1 = Pe{1,cl}(i,k)*Qe{1,cl}(k,j);
                        e2 = Pe{1,cl}(i,k)^2 + Qe{1,cl}(k,j)^2;
                    end
                    e = e + sqrt(( Riu(i,j)-e1 )^2 + (beta/2) * e2);               
                end
                
            end
        end
        
        if abs(e) < 0.01
            break
        end
    end
end



end