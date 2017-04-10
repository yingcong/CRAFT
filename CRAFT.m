% Transform feasture (either in the original space or in the kernel space) to the augmented
% space by CRAFT
% input:
% X: column-wise feature (Dimension * nSample)
% view_label:  integer vector whose elements are the view labels
% (i.e., from which camera the image sample is collected) of each sample.
% omega: the camera correlation matrix. It is generated by cameraCorrelation.m 
% beta: the Camera View Discrepancy Regularization term. It should be (0,1)
% K: the Kernel matrix of the training set. This is useful when the subspace learning is conducted
% in the kernel space. When it is conducted in the original space, we set K=eye(Dimension)
% Output:
% Y: the transformed feature.
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
function Y = CRAFT(X,view_label,omega,beta,K)
view_label_uniq = unique(view_label);
nViews = numel(view_label_uniq);


Y = [];

for i = 1 : numel(view_label_uniq)
    Y = [Y;X];
end

d = size(X,1);
for i = 1 : numel(view_label_uniq)
    for j = 1 : numel(view_label_uniq)
        whos
        Y(1+(i-1)*d : i*d,view_label == view_label_uniq(j))...
        = Y(1+(i-1)*d : i*d,view_label == view_label_uniq(j)) * omega(i,j);
    end
end


I = eye(d);
for i = 1 : numel(view_label_uniq)
    for j =1 : numel(view_label_uniq)
        if i == j
            C{i,j} = K;
        else
            C{i,j} = -beta  * K;
        end
    end
end
C = cell2mat(C);
[P,Lambda] = eig(C);
Y = Lambda^(-0.5)*P'*Y;
end


