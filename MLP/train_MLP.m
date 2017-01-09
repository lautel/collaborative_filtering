function [mlp,mse_r] = train_MLP(mlp, X, T, n_epochs)

n_samples = size(X,2);


for r=1:n_epochs

    fprintf(1,'Epoch %d / %d\n', r, n_epochs);
    mse = 0.0;
    ce  = 0.0;
    n   = 1;    
    while n<n_samples - mlp.B          
        n0 = n;
        n1 = min(n0 + mlp.B - 1, n_samples);
             
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
        
        
        learnrate = mlp.learnrate/mlp.B;
        decay     = mlp.decay;
        
        % Backward-Pass --------------------------------------------
        for i=mlp.n_layers:-1:1        
            
            mlp.layer(i).D = mlp.wlayer(i).W * mlp.layer(i+1).D;
              
            mlp.layer(i).D = mlp.layer(i).D .* ( 1 - (mlp.layer(i).A).^2 );
                 
            

            dW =  mlp.layer(i).A * mlp.layer(i+1).D';
       
            mlp.wlayer(i).W = (1-decay*learnrate) * mlp.wlayer(i).W + learnrate * dW;  

            db = sum(mlp.layer(i+1).D, 2);

            mlp.wlayer(i).b = (1-decay*learnrate) * mlp.wlayer(i).b + learnrate * db;
        end
        
        n = n + n1 - n0 + 1;
        
    end   
        
    fprintf(1,'MSE: %f\n', (mse / n_samples));           
%     fprintf(1,'Classification Error Train (%d/%d) %d: %.2f %%\n', ...
%                     ce,n_samples,mlp.derror, (ce / n_samples)*100 );       
    mse_r(r) = (mse/n_samples);
end


%------------------------------------------------------------------

function [mlp, mse, ce] = set_tgt( mlp , T)  

        if(mlp.derror == -1)
            derror = mlp.layers(end);
        else
            derror = mlp.derror;
        end               
               
        % deltas = teacher - activations
        mlp.layer(end).D = T - mlp.layer(end).A;
        
        mse = sum ( sum( mlp.layer(end).D.^2 ) );
        
        % ce , num errors        
        
        [m0, im0] = max( T(1:derror,:) );
        [m1, im1] = max( mlp.layer(end).A(1:derror,:) );
        
        ce = length( find( im0 ~= im1 ) );

        


      
  
