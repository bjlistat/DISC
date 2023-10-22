function [criteria,idx,Y,U] = DISC_NT(X,n_class,opts)

% Deep Inteligent Spectral Clustering with multiple layers

ab = opts.ab;
labels = opts.labels;
distmetric = opts.distmetric;
n_layer = opts.layers;
s_class = opts.classsize; 

k_0 = 5; if isfield(opts,'k0'), k_0 = opts.k0; end
repa = 1; if isfield(opts,'repa'), repa = opts.repa; end




num_eig = n_class + 5; if isfield(opts,'num_eig'), num_eig = opts.num_eig; end


n = size(X,2); 

alpha = 6; if n/n_class<=100, alpha = 9; end  
if isfield(opts,'alpha'), alpha = opts.alpha; end

% Step 1. Fisrt layer of DISC
% 1.1 Estimated class-consistent NB of X
[nb_idx,nb_dist,nb_0] = nbhood_X(X,n_class,k_0,distmetric); %ts(1) = toc

% 1.2 Probobility graph of the estimated class-consistent NB
A = sigmoid_graph_X(nb_idx,nb_dist,alpha);  

% 1.3 Ncut on A
d = sqrt(sum(A,2));
D_inv_sqrt = spdiags(1./d,0,n,n);
G = speye(n)-D_inv_sqrt*A*D_inv_sqrt; clear A
G = (G+G')/2;

if n<1000
  [U,~] = eigs(G, num_eig,'smallestabs','SubspaceDimension',min(n,1000)); 
else
  [U,~] = eigs(G, num_eig,'smallestabs');
end
clear G

maxu = max(abs(U),[],1);

if all(maxu(1:n_class)<=0.5) %no isolate points in the first n_class eigenvectors.
    U = U(:,1:n_class);
else
    ii = nnz(maxu>0.5);
    U = U(:,1: min(n_class + ii, num_eig)); %the column should be no more than n_class + 5.
end

clear G

% 1.4 Estimate the class-consistent neighbor sets of Y = U'
[I,J,nb_sdist,sigma] = nbhood_U(U,n_class,s_class,k_0); 

% Step 2. First iteration of the Intelligent Spectral Clustering
func = opts.func;

[idx,Y,phi] = ISC(ab,I,J,nb_sdist,sigma,nb_0,n_class,s_class,n_layer,func, repa);


acmiari = [calAC(idx,labels) calMI(idx,labels) calARI(idx,labels)];
[sc,db,ch] = InteriorIndices(Y',idx);

fprintf(' Initial  phi = %4.5e, ac = %4.2f, mi = %4.2f, ari = %4.2f,',phi,acmiari*100);
fprintf(' sc: %4.2f db: %4.2f ch: %4.2e',sc,db,ch)
fprintf('\n')

criteria = [phi acmiari sc db ch]; 