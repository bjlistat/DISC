addpath(genpath('Data'));
addpath(genpath('Tools'));
rng default;


Datasets = {'CP'; 'PB'; 'SP'}; n_datasets = size(Datasets,1);
nmlzs = [2,2,2];

for id = 1:n_datasets

    data = Datasets{id};
    isnormal = nmlzs(id);
    eval(['load ' data])
    X = fea';
    labels = gnd-min(gnd)+1;
    n_class = max(gnd)-min(gnd)+1;
    m_ = ceil(length(labels)/n_class);

    if isnormal == 0
        dist_type = 'euclidean';
    elseif isnormal == 1
        dist_type = 'nm_seuclidean';
    elseif isnormal == 2
        dist_type = 'cityblock';
    elseif isnormal == 3
        dist_type = 'spearman';
    end

    n_layers = 2;
    fprintf('n_layer: %2d\n',n_layers)
    ab = [2*ones(n_layers,1),[1;zeros(n_layers-1,1)]];
    opts.ab = ab;
    opts.labels = labels;
    opts.distmetric = dist_type;
    opts.layers = 2;
    opts.MaxIter = 5;
    opts.classsize = m_;
    opts.func = 'db/(sc*ch)';
    opts.repa = 1;
    [criteria,Idx] = DISC_NT(X,n_class,opts);
    pred_groups = Idx(:,end);    

end
