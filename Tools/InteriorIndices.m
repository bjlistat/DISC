function [sc,db,ch] = InteriorIndices(Y,idx)

% ItrIdx returns the means of the interior indices:
%   sc: silhouette coefficient, 
%       the ralatie gap of intra-class and inter-class distances
%   db: Davies-Bouldin criterion, 
%   ch: Calinski-Harabasz index
%   dn: Dunn index, 
%       the ratio of minimal inter-class distanc and intra-class distance
%   sd: squared distance to the class centriod

% Each row of Y is a point. 

if 0

  [n,m] = size(Y); K = max(idx);
  c_Y = mean(Y,1);

  c = zeros(K,m);
  si = zeros(K,1);
  sd = zeros(K,1);
  ad = zeros(K,1);
  nc = zeros(K,1);
  for k=1:K
    Yk = Y(idx==k,:); nc(k) = size(Yk,1);
    c(k,:) = mean(Yk,1);
    
    dk = pdist2(Yk,c(k,:));
    ad(k) = mean(dk);
    sd(k) = sum(dk.^2);

    ak = sum(pdist2(Yk,Yk),2)/(nc(k)-1);
    bk = inf;
    for ell=1:K
      if ell==k, continue, end
      bk = min(bk,mean(pdist2(Yk,Y(idx==ell,:)),2));
    end
    si(k) = sum((bk-ak)./(max(ak,bk)));
  end
  sc = sum(si)/n;

  dcc = pdist2(c,c);
  S = ad+ad'; S = S-diag(diag(S));
  db = mean(max(S./(dcc+eye(K)),[],2));

  dcY = pdist2(c,c_Y);
  SS_b = sum(nc.*(dcY.^2))/(K-1);
  SS_w = sum(sd)/(n-K);
  ch = SS_b/SS_w;
else
  eva = evalclusters(Y,idx,'CalinskiHarabasz'); 
  ch = eva.CriterionValues;
  eva = evalclusters(Y,idx,'DaviesBouldin'); 
  db = eva.CriterionValues;
  eva = evalclusters(Y,idx,'silhouette'); 
  sc = eva.CriterionValues;
end





