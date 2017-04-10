% Estimate the camera correlation. It is used in CRAFT.m
% input:
% X: the column-wise feature in the original feature space(Dimension * nSample). 
% view_label:  integer vector whose elements are the view labels
% (i.e., from which camera the image sample is collected) of each sample.
% output:
% omega: the camera correlation matrix (nView * nView).
% 
% If you have any questions/suggestions regarding the code or the paper, 
% please feel free to contact me.  Email: yingcong.ian.chen@gmail.com
% Ying-Cong Chen, 1/24/2017
%
% ----------------------------------------------------------------------------
% Please kindly cite our paper if you find this code is useful:              %
% Ying-Cong Chen, Xiatian Zhu, Wei-Shi Zheng, and Jianhuang Lai,             %
% Person Re-Identification by Camera Correlation Aware Feature Augmentation  %
% Pattern Analaysis and Machine Intelligence, 2017                           %
% ----------------------------------------------------------------------------

function omega = cameraCorrelation(X,view_label)
nViews = numel(unique(view_label));
nSample = 100;
views = 1:nViews;
for i = 1 : nViews
    for j = 1 : nViews
        X1 = X(:,view_label == views(i));
        X2 = X(:,view_label == views(j));
        n1 = size(X1,2);
        n2 = size(X2,2);
        if n1 < 2 || n2 < 2
            omega(i,j) = 1;
            continue;
        end
        if i ~= j
            [omega(i,j)] = princAngle(X1,X2,nSample);
        else
            omega(i,j) = 1;
        end
    end
end
omega = 0.5 * (omega+omega');
for i = 1 : nViews
    s = 0;
    for  j = 1 : nViews
        if i ~= j
            s = s + omega(j,i);
        end
    end
    omega(i,i) = 2 - s/(nViews-1);
end
omega = normc(omega);
end

% for acceleration, we randomly select n sample to estimate the principle angles
function [r] = princAngle(X1,X2,nSample)
rng(1)
p1 = randperm(size(X1,2));
p2 = randperm(size(X2,2));
np1 = numel(p1);
np2 = numel(p2);
p1 = p1(1:min(np1,nSample));
p2 = p2(1:min(np2,nSample));

if exist('pca') % for matlab 2012 or above
    G1 = pca(X1(:,p1)');
    G2 = pca(X2(:,p2)');
else % for early version
    G1 = princomp(X1(:,p1)','econ');
    G2 = princomp(X2(:,p2)','econ');
end

cos_ = svd(G1'*G2);
r = mean(cos_);
end