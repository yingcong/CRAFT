% The MFA distance matrix learning. The original version is implemented in:
% Feng Xiong, et.al. Person Re-Identification using Kernel-based Metric Learning Methods
% ECCV 2014
classdef MFA_class < handle
    properties
        special = {};
        Kdist = [];
        alpha = 1e-8;
        rank = 100;
        Nw = 10;
        Nb = 30;
        projMat = [];
    end
    
    methods
        function setPara( this, conf )
            param = fields(conf);
            for i = 1:numel(param)
                field = param{i};
                if any( strcmp(properties(this), field) )
                    this.(field) = conf.(field);
                end
            end
        end
        function [Method, V]= MFA(this,X, id, option)
            % implemented according to  CVPR2013: Graph Embedding and Extensions: A General
            % Framework for Dimensionality Reduction
            % By Fei Xiong,
            %    ECE Dept,
            %    Northeastern University
            %    2014-02-15
            % INPUT
            %   X: N-by-d data matrix. Each row is a sample vector.
            %   id: (N-by-1) the identification number for each sample
            %   option: algorithm options
            %       beta: the parameter of regularizing S_b
            %       d: the dimensionality of the projected feature
            %       eps: the tolerence value.
            %       NNw: number of samples used for within class (0 for maximum possible)
            %       NNb: number of samples used for between class
            % Note that the kernel trick is used here.
            % OUTPUT
            %   Method: the structure contains the learned projection matrix and
            %       algorithm parameters setup.
            %   V: the eigenvalues
            T =[];
            V =[];
            %             display(['begin MFA ' option.kernel]);
            beta= option.beta;
            d = option.d;
            Method = struct('rbf_sigma',0);
            %             [K, Method] = this.ComputeKernel(X, 'linear', Method);
            [Ww, Wb] = this.MFAAffinityMatrix(this.Kdist, id, option.Nw,option.Nb); % compute W, Wp in equation 13 and 14
            Ew = diag(sum(Ww)) - Ww;
            Eb = diag(sum(Wb)) - Wb;
            
            Sw = X'*Ew*X; % equation 13
            Sb = X'*Eb*X; % equation 14
            Sw =(1-beta)*Sw + beta*trace(Sw)*eye(size(Sw))/size(Sw,1);
            %             [T, V]= eig(Sb, Sw);
            %             lambda = diag(V);
            %             [~,idx] = sort(lambda,'descend');
            %             T = T(:,idx(1:d));
            if d > 0
                [T, V]= eigs(Sb, Sw, d);
            else
                [T, V]= eig(Sb, Sw);
                lambda = diag(V);
                [~,idx] = sort(lambda,'descend');
                T = T(:,idx(1:end));
            end
            options.intraK = option.Nw;
            % options.interK = option.Nb;
            % options.Regu = 1;
            % options.ReducedDim = d;
            % options.ReguAlpha = 0.1;
            % [T, ~] = MFA_CDeng(id, options, K');
            % T = T';
            
            Method.name = 'MFA';
            Method.P=T;
            Method.kernel=option.kernel;
            return;
        end
    end
    methods (Static)
        function [Ww, Wb] = MFAAffinityMatrix(K, id, NNw,NNb)
            % compute distance in the kernel space using kernel matrix
            temp = repmat(diag(K), 1, size(K,1));
            dis = temp + temp' - 2*K;
            dis(sub2ind(size(dis), [1:size(dis,1)], [1:size(dis,1)]))=inf;
            temp = repmat(id.^2, 1, length(id));
            idm = temp + temp' - 2*id*id';
            
            disw = dis;
            disw(idm~=0) = inf;
            [temp, ixw]= sort(disw);
            ixw(isinf(temp))=0;
            
            if NNw==0 % Use the maximum possible number of within class
                NNw = min(sum(~isinf(temp)));
            end
            
            ixw = ixw(1:NNw, :);
            ixtmp = repmat([1:size(K,1)], NNw, 1);
            ixtmp= ixtmp(ixw(:)>0);
            ixw = ixw(ixw(:)>0);
            ixtmp = sub2ind(size(K), ixtmp(:), ixw(:));
            Ww = zeros(size(K));
            Ww(ixtmp) = 1;
            Ww = Ww+ Ww';
            Ww = double(Ww>0);
            
            disb = dis;
            disb(idm==0) = inf;
            [temp, ixb]= sort(disb);
            ixb(isinf(temp))=0;
            ixb = ixb(1:NNb, :);
            ixtmp = repmat([1:size(K,1)], NNb, 1);
            ixtmp= ixtmp(ixb(:)>0);
            ixb = ixb(ixb(:)>0);
            ixtmp = sub2ind(size(K), ixtmp(:), ixb(:));
            ixtmp= ixtmp(ixtmp>0);
            Wb = zeros(size(K));
            Wb(ixtmp) =1;
            Wb = Wb+ Wb';
            Wb = double(Wb>0);
            return;
        end
        % equation 1 and 2 from paper: "Self-Tuning Spectral Clustering"
        % INPUT
        %   K: KernelMatrix
        %   NN: the index of the nearest neighbors used to scaled the distance.
        % OUTPUT
        %   A: Affinity matrix
        function [A] = LocalScalingAffinity(K, NN)
            
            % compute distance in the kernel space using kernel matrix
            temp = repmat(diag(K), 1, size(K,1));
            dis = temp + temp' - 2*K;
            
            [disK, ~]= sort(dis);
            disK = sqrt(disK(NN+1,:));
            disK = disK' * disK;
            A = exp(-(dis./disK));
            A = A-diag(diag(A));
            return;
        end
        % calculate the kernel matrix.
        % By Fei Xiong,
        %    ECE Dept,
        %    Northeastern University
        %    2013-11-04
        % Input:
        %       Method: the distance learning algorithm struct. In this function
        %               two field are used.
        %               rbf_sigma is written while computing the rbf-chi2 kernel
        %               matrix.
        %               kernel is the name of the kernel function.
        %       X:  Each row is a sample vector. N-by-d
        
        
        function [K, Method] = ComputeKernel(X, kernel, Method)
            K= single(zeros(size(X,1)));
            if (size(X,2))>2e4 && (strcmp(kernel, 'chi2') || strcmp(kernel, 'chi2-rbf'))
                matlabpool open
                switch kernel
                    case {'linear'}% linear kernel
                        K = X*X';
                    case {'chi2'}% chi2 kernel
                        parfor i =1:size(X,1)
                            dotp = bsxfun(@times, X(i,:), X);
                            sump = bsxfun(@plus, X(i,:), X);
                            K(i,:) = full(2* sum(dotp./(sump+1e-10),2));
                        end
                        clear subp sump;
                    case {'chi2-rbf'}% chi2 RBF kernel
                        parfor i =1:size(X,1)
                            subp = bsxfun(@minus, X(i,:), X);
                            subp = subp.^2;
                            sump = bsxfun(@plus, X(i,:), X);
                            K(i,:) = full(sum(subp./(sump+eps),2));
                        end
                        temp = triu(ones(size(K))-eye(size(K)))>0;
                        temp = K(temp(:));
                        [temp,~]= sort(temp);
                        % rbf-chi2 kernel parameter can be set here. For example, the
                        % first quartile of the distance can be used to normalize the
                        % distribution of the chi2 distance
                        Method.rbf_sigma = 1; %temp(ceil(length(temp)*0.25));
                        K =exp( -K/Method.rbf_sigma);
                        clear subp sump;
                end
                matlabpool close;
            else
                switch kernel
                    case {'linear'}% linear kernel
                        K = X*X';
                    case {'chi2'}% chi2 kernel
                        for i =1:size(X,1)
                            dotp = bsxfun(@times, X(i,:), X);
                            sump = bsxfun(@plus, X(i,:), X);
                            K(i,:) = full(2* sum(dotp./(sump+1e-10),2));
                        end
                        clear subp sump;
                    case {'chi2-rbf'}% chi2 RBF kernel
                        for i =1:size(X,1)
                            subp = bsxfun(@minus, X(i,:), X);
                            subp = subp.^2;
                            sump = bsxfun(@plus, X(i,:), X);
                            K(i,:) = full(sum(subp./(sump+1e-10),2));
                        end
                        temp = triu(ones(size(K))-eye(size(K)))>0;
                        temp = K(temp(:));
                        [temp,~]= sort(temp);
                        Method.rbf_sigma = 1;
                        K =exp( -K/Method.rbf_sigma);
                        clear subp sump;
                end
            end
            return;
        end
        
    end
    
    methods
        
        function this = KMFA2_class()
        end
        function train(this,X,group)
            group = group(:);
            MFA_opts.beta = this.alpha;
            MFA_opts.Regu = 1;
            MFA_opts.kernel = 'linear';
            MFA_opts.d = this.rank;
            MFA_opts.Nw = this.Nw;
            MFA_opts.Nb = this.Nb;
            [Method,~] = this.MFA(X',group,MFA_opts);
            this.projMat = Method.P;
        end
        
        function Z = calDistMat(this,X1,X2)
            M = this.projMat*this.projMat';
            Z = this.metricDist(X1,X2,M);
        end
        function Z = metricDist(this,X,Y,M)
            [~,nX]=size(X);
            [~,nY]=size(Y);
            XMX = diag(X'*M*X);
            YMY = diag(Y'*M*Y);
            XMY = X'*M*Y;
            XMX_Mat = repmat(XMX,[1,nY]);
            YMY_Mat = repmat(YMY',[nX,1]);
            Z = XMX_Mat + YMY_Mat - 2*XMY;
        end
    end
end