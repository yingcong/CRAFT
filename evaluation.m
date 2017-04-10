% evaluating the performance
% input:
% Xtrain, Xtest: the column-wise feature matrix (Dimension * nSample)
% LabelTrain, LabelTest: integer vectors whose elements are identity labels of each sample
% ViewTrain,ViewTest: integer vectors whose elements are the view labels
% (i.e., from which camera the image sample is collected) of each sample.
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
function [CMC]=evaluation(Xtrain,Xtest,LabelTrain,LabelTest,ViewTrain,ViewTest,opt)

LabelTest1 = LabelTest(ViewTest==1);
LabelTest2 = LabelTest(ViewTest==2);
% transform the feature to the kernel space
Ktrain = batt_kernel(Xtrain,Xtrain);
Ktest = batt_kernel(Xtrain,Xtest);
Kmat = Ktrain;
if opt.useCRAFT==1
	beta = 0.6;
	if isfield(opt,'beta')
		beta = opt.beta;
	end
	% estimate the camera correlation
	omega = cameraCorrelation(Xtrain,ViewTrain); 
	% perform CRAFT transformation
	Ktrain = CRAFT(Ktrain,ViewTrain,omega,beta,Kmat);
	Ktest = CRAFT(Ktest,ViewTrain,omega,beta,Kmat);
end
P = MFA_class();
opts.method_opts.Kdist = Kmat;
P.setPara(opts.method_opts);
P.train(Ktrain,LabelTrain);
% testing
Ktest1 = Ktest(:,ViewTest==1);
Ktest2 = Ktest(:,ViewTest==2);
distMat = P.calDistMat(Ktest1,Ktest2);
CMC = getCMC(distMat,LabelTest1,LabelTest2);
end

function Z = batt_kernel(X,Y)
Z = sqrt(X)' * sqrt(Y);
end
function Z = linear(X,Y)
Z = (X)' * (Y);
end