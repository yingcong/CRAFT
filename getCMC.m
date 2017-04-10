% compute CMC. It is compatible with both single-shot scenario and multi-shot scenario.
% For multi-shot, average distance of the image pairs is used as the distance between different
% identities.
%
% input:
% distMat: the distance matrix (nGalleryImage * nProbeImage). Larger value means less similar
% label_gallery: the label of the gallery set. 
% label_probe: the label of the probe set.
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
function CMC = getCMC(distMat,label_gallery,label_probe)
label_gallery = label_gallery(:);
label_probe = label_probe(:);
ER=[];
unique_gallery= unique(label_gallery);
unique_probe = unique(label_probe);
for i = 1 : numel(unique_gallery)
    for j = 1 : numel(unique_probe)
        cellDist{i,j} = distMat(label_gallery == unique_gallery(i), label_probe == unique_probe(j));
    end
end

for i = 1 : numel(unique_gallery)
    for j = 1 : numel(unique_probe)
        tmp = cellDist{i,j};
        dist(i,j) = mean(tmp(:));
    end
end
TrueTable_gallery=repmat(unique_gallery,[1,numel(unique_probe)]);
TrueTable_probe=repmat(unique_probe',[numel(unique_gallery),1]);
TrueTable=(TrueTable_gallery==TrueTable_probe);


for rank=1:numel(unique_gallery)
    errorNum=0;
    for i=1:numel(unique_probe) 
        dis=dist(:,i); 
        [~,index]=sort(dis,'ascend');
        index=index(1:rank);
        if sum(TrueTable(i,index))==0
            errorNum=errorNum+1;
        end
    end
    ER=[ER errorNum/numel(unique_probe) ];
end

CMC = 1 - ER;
end