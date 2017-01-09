function [Pe, Qe] = my_svd_uno(Pe, Qe, Riu, R, N, M, K, alpha, beta)
% SVD para el conjunto entero de usuarios

Qe = Qe.';

J=0; 

    for step=1:100      % Número de iteraciones
        fprintf('Iteration n = %d /100 \n',step);
        for i=1:N       % Películas
            for j=1:M   % Usuarios
                
                if ~isnan(Riu(i,j))
                    eij = R(i,j) * (Riu(i,j) - Pe(i,:)*Qe(:,j));
                    for k=1:K
                        Pe(i,k) = Pe(i,k) + alpha * (2 * eij * Qe(k,j) - beta * Pe(i,k));
                        Qe(k,j) = Qe(k,j) + alpha * (2 * eij * Pe(i,k) - beta * Qe(k,j));
                    end
                end
                
            end
        end
                
        
        % Actualización función de coste

        for i=1:N      % Películas
            for j=1:M  % Usuarios
                
                if ~isnan(Riu(i,j))
                    for k=1:K
                        e1 = Pe(i,k)*Qe(k,j);
                        e2 = Pe(i,k)^2 + Qe(k,j)^2;
                    end
                    J = J + sqrt(( Riu(i,j)-e1 )^2 + (beta/2) * e2);   
                    
                end
                
            end
        end
      
        if abs(J) < 0.01
            break
        end
    end
    
    Qe = Qe.';
end