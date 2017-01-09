function [er, mse, O]= test_MLP(mlp, X, T)

n_samples = size(X,2);


mse = 0.0;
ce  = 0.0;
n   = 1;    
while n<=n_samples - mlp.B +1          
    n0 = n;
    n1 = min(n0 + mlp.B - 1, n_samples);
    
    %[n0,n1]
 
    mlp.layer(1).A = X(:,n0:n1);
    Tb             = T(:,n0:n1);
    
    % Forward-Pass --------------------------------------------
    for i=1:mlp.n_layers
         %mlp = forward( mlp, i );
        
         mlp.layer(i+1).A = tanh(                                   ...
                                mlp.wlayer(i).W' * mlp.layer(i).A   ...
                                +                                   ...
                                mlp.wlayer(i).b  * ones(1,mlp.B)    ...
                                );
                                              

        
    end
    % Forward-Pass --------------------------------------------
    
    
    %calculate error at output layer -> teacher
    
    [mlp, mseb, ceb] = set_tgt( mlp , Tb);
    
    mse = mse + mseb;
    ce  = ce  + ceb;
  
    
	O(:,n0:n1) = mlp.layer(end).A;
    
    n = n + n1 - n0 + 1;
        
    end
        
    fprintf(1,'MSE: %f\n', (mse / n_samples));           
    fprintf(1,'Classification Error Test (%d/%d) %d: %.2f %%\n', ...
                    ce,n_samples,mlp.derror, (ce / n_samples)*100 );  
    
    er = (ce / n_samples)*100;


%------------------------------------------------------------------

function [mlp, mse, ce] = set_tgt( mlp , T)  

        if(mlp.derror == -1)
            derror = mlp.layers(end);
        else
            derror = mlp.derror;
        end               
               
        % deltas = teacher - activations
        mlp.layer(end).D = T - mlp.layer(end).A;
        
        % mse = sum( d.^2 )
        %mse = 1/2 * sum ( sum( mlp.layer(end).D.^2 ) );% / mlp.B;
        mse = sum ( sum( mlp.layer(end).D.^2 ) );
        
        % ce , num errors        
        
        [m0, im0] = max( T(1:derror,:) );
        [m1, im1] = max( mlp.layer(end).A(1:derror,:) );
        
        ce = length( find( im0 ~= im1 ) );

      
  
