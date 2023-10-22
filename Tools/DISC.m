function [criteria,Idx,Y] = DISC(X,n_class,opts)

% Deep Inteligent Spectral Clustering with multiple layers

ab = opts.ab;
MaxIter = opts.MaxIter;
labels = opts.labels;
distmetric = opts.distmetric;
n_layer = opts.layers;
s_class = opts.classsize; 

k_0 = 5; if isfield(opts,'k0'), k_0 = opts.k0; end
repa = 1; if isfield(opts,'repa'), repa = opts.repa; end

num_eig = n_class + 5; if isfield(opts,'num_eig'), num_eig = opts.num_eig; end


n = size(X,2); 

%tic;
% Step 1. Fisrt layer of DISC
% 1.1 Estimated class-consistent NB of X
[nb_idx,nb_dist,nb_0] = nbhood_X(X,n_class,k_0,distmetric); %ts(1) = toc

% 1.2 Probobility graph of the estimated class-consistent NB
alpha = 6; if n/n_class<=100, alpha = 9; end
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
[I,J,nb_sdist,sigma] = nbhood_U(U,n_class,s_class,k_0); %ts(3) = toc

% Step 2. First iteration of the Intelligent Spectral Clustering

if MaxIter == 0
  [Idx,Y] = ISC(ab,I,J,nb_sdist,sigma,nb_0,n_class,s_class,n_layer, repa);
  criteria  = [calAC(Idx,labels) calMI(Idx,labels) calARI(Idx,labels)];
  return
end

func = opts.func;
[idx,Y,phi] = ISC(ab,I,J,nb_sdist,sigma,nb_0,n_class,s_class,n_layer,func, repa);
acmiari = [calAC(idx,labels) calMI(idx,labels) calARI(idx,labels)];
[sc,db,ch] = InteriorIndices(Y',idx);

fprintf(' Initial  phi = %4.5e, ac = %4.2f, mi = %4.2f, ari = %4.2f,',phi,acmiari*100);
fprintf(' sc: %4.2f db: %4.2f ch: %4.2e',sc,db,ch)
fprintf('\n')

% Step 3. Random optimization

delta = repmat([0.2 0.2],n_layer,1); 
delta(1,:) = 0; delta(end,2) = 0;

criteria = zeros(MaxIter,7); 
criteria(1,:) = [phi acmiari sc db ch]; 
Idx = zeros(n,MaxIter); Idx(:,1) = idx; 

dphi = 1; iter = 1; 

while any(abs(delta(:))>=0.01) && iter<=MaxIter && abs(dphi)>1.e-4
  for t=2:n_layer
    for i=1:2
      if (t==n_layer && i==2) || abs(delta(t,i))<0.01
        continue
      end
      ab_ = ab;
      if i==1, ab_(t,i) = min(3,max(ab(t,i)+delta(t,i),1)); end
      if i==2, ab_(t,i) = min(1,max(ab(t,i)+delta(t,i),0)); end
      
      if ab_(t,i)==ab(t,i), delta(t,i) = 0; continue, end
      [idx_,Y_,phi_] = ISC(ab_,I,J,nb_sdist,sigma,nb_0,n_class,s_class,n_layer,func, repa);
      
      % Check convergence
      dphi = (phi-phi_)/phi;
      if phi_<phi
        iter = iter+1;        
        phi = phi_; idx = idx_; ab = ab_; Y = Y_;
        acmiari = [calAC(idx,labels) calMI(idx,labels) calARI(idx,labels)];
        [sc,db,ch] = InteriorIndices(Y',idx);    
        
        criteria(iter,:) = [phi acmiari sc db ch]; 
        Idx(:,iter) = idx;

        delta(t,i) = delta(t,i)*2;
      else
        delta(t,i) = -delta(t,i)/2;
      end
    end
  end
end
fprintf(' Final    phi = %4.5e, ac = %4.2f, mi = %4.2f, ari = %4.2f,',phi,acmiari*100)
fprintf(' sc: %4.2f db: %4.2f ch: %4.2e,',sc,db,ch)
fprintf(' ab(%1d,%1d) = %4.2f, delta_ti = %4.2f',t,i,ab_(t,i),delta(t,i))
fprintf('\n')
criteria(iter+1:end,:) = []; Idx(:,iter+1:end) = [];
