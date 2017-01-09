function mlp = initiate_MLP(layers, B)

mlp.layers   = layers;
mlp.B        = B;
mlp.n_layers = length(mlp.layers) - 1;

mlp.learnrate = 0.10;
mlp.decay     = 0.01;
mlp.derror    = -1;

for i=1:mlp.n_layers+1

    mlp.layer(i).A = zeros( mlp.layers(i), B);
    mlp.layer(i).D = zeros( mlp.layers(i), B);

end

for i=1:mlp.n_layers
    
    mlp.wlayer(i).W = rand( mlp.layers(i), mlp.layers(i+1) );
    mlp.wlayer(i).W = mlp.wlayer(i).W - 0.5;
    mlp.wlayer(i).W = mlp.wlayer(i).W / 10.0;
    mlp.wlayer(i).b = zeros(mlp.layers(i+1),1);
    
end
