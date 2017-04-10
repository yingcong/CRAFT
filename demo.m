% This demo represents the performance of HIPHOP feature 
% and the effectiveness of CRAFT transformation
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

clear,clc,close all
dataset = 'viper';
sep = 'train_test_split';
load(dataset);
load(sep);
opt.useCRAFT = 0;
disp('Testing HIPHOP feature without CRAFT.');
parfor i = 1 : 10
    trainIdx = ismember(Label,trainPerson{i});
    testIdx = ~trainIdx;
    Xtrain = X(:,trainIdx);
    Xtest = X(:,testIdx);
    LabelTrain = Label(trainIdx);
    LabelTest = Label(testIdx);
    ViewTrain = ViewLabel(trainIdx);
    ViewTest = ViewLabel(testIdx);
    CMC_woCRAFT(i,:) = evaluation(Xtrain,Xtest,LabelTrain,LabelTest,ViewTrain,ViewTest,opt);
end
CMC_woCRAFT = mean(CMC_woCRAFT,1);
disp('Rank1   Rank5   Rank10   Rank20')
CMC_woCRAFT = CMC_woCRAFT*100;
fprintf('%2.1f,   %2.1f,   %2.1f,   %2.1f\n',CMC_woCRAFT(1),CMC_woCRAFT(5),CMC_woCRAFT(10),CMC_woCRAFT(20));


disp('Testing HIPHOP feature + CRAFT');
opt.useCRAFT = 1;
opt.beta = 0.6;
parfor i = 1 : 10
    trainIdx = ismember(Label,trainPerson{i});
    testIdx = ~trainIdx;
    Xtrain = X(:,trainIdx);
    Xtest = X(:,testIdx);
    LabelTrain = Label(trainIdx);
    LabelTest = Label(testIdx);
    ViewTrain = ViewLabel(trainIdx);
    ViewTest = ViewLabel(testIdx);
    CMC_CRAFT(i,:) = evaluation(Xtrain,Xtest,LabelTrain,LabelTest,ViewTrain,ViewTest,opt);
end
CMC_CRAFT = mean(CMC_CRAFT,1);
disp('Rank1   Rank5   Rank10   Rank20')
CMC_CRAFT = CMC_CRAFT*100;
fprintf('%2.1f,   %2.1f,   %2.1f,   %2.1f\n',CMC_CRAFT(1),CMC_CRAFT(5),CMC_CRAFT(10),CMC_CRAFT(20));