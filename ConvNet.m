classdef ConvNet < handle
    
    properties (SetAccess = private)
        net;
        theta;
        dtheta
        nLayers;
        O;
        delta;
        lossInd;
        Loss;
        AllLoss;
    end
    properties (SetAccess = private, GetAccess = private)
        mygpuArray;
        mygather;
        printDecay;
        printIter;
        snapshotFile;
    end
    
    
    methods
        
        function this = ConvNet(netD,atGPU,weightsInitialization)
            % Constructor. See demoMNIST for example
            if nargin<3
                weightsInitialization = 'Orthogonal';
            end
            if atGPU
                this.mygpuArray = @(x) gpuArray(x);
                this.mygather = @(x) gather(x);
            else
                this.mygpuArray = @(x) x;
                this.mygather = @(x) x;
            end
            netD=this.add_reduced_conv_layers(netD);
            this.initializeNet(netD,atGPU);
            this.initializeWeights(weightsInitialization);
            this.nLayers = length(this.net);
            this.delta = cell(size(this.O));
            % make everything single and atGPU
            this.theta = this.mygpuArray(single(this.theta));
            for i=1:length(this.O)
                if ~isempty(this.O{i})
                    this.O{i} = this.mygpuArray(this.O{i});
                end
            end

        end
        
        % forward function
        function forward(this,I)
            
            for i=1:this.nLayers
                outInd = this.net{i}.outInd;
                inInd = this.net{i}.inInd;
                switch this.net{i}.type
                    case 'input'
                        this.O{outInd} = this.net{i}.data.get(I);
                    case 'duplicate'
                        for j=outInd
                            this.O{j} = this.O{inInd};
                        end
                    case 'im2col'
                        this.O{outInd} = this.O{inInd}(this.net{i}.im2colInd); % fast im2col
                    case 'max'
                        this.O{outInd} = max(this.O{inInd});
                    case 'mean'
                        this.O{outInd} = mean(this.O{inInd});
                    case 'reshape'
                        this.O{outInd} = reshape(this.O{inInd},this.net{i}.newshape);
                    case 'permute'
                        this.O{outInd} = permute(this.O{inInd},this.net{i}.newshape);
                    case 'affine'
                        this.O{outInd} = reshape(this.theta(this.net{i}.Ws:this.net{i}.We),this.net{i}.Wshape)*this.O{inInd} + this.theta(this.net{i}.bs:this.net{i}.be)*this.net{i}.ones;
                    case 'relu'
                        this.O{outInd} = max(0,this.O{inInd});
                    case 'clamp'
                        this.O{outInd} = min(1,max(0,this.O{inInd}));
                    case 'loss'
                        this.O{outInd} = this.net{i}.loss.LossAndErr(this.O(inInd));
                    case 'concat'
                        this.O{outInd} = cat(this.net{i}.dim,this.O{inInd});
                    case 'split'
                        tmp = cell(length(size(this.O{inInd})),1);
                        for j=1:length(tmp), tmp{j} = ':'; end;
                        pos=0; nl = size(this.O{outInd(1)},this.net{i}.dim);
                        for j=1:length(outInd)
                            tmp{this.net{i}.dim} = pos + (1:nl);
                            pos = pos+nl;
                            this.O{outInd(j)} = this.O{inInd}(tmp{:});
                        end
                    case 'pad'
                        blobSize = size(this.O{inInd});
                        this.O{outInd}(1:blobSize(1),1:blobSize(2),:,:) = this.O{inInd};
                    case 'elementwiseProd'
                        this.O{outInd} = this.O{inInd(1)} .* this.O{inInd(2)};
                    case 'add'
                        this.O{outInd} = this.net{i}.alpha * this.O{inInd(1)} + this.net{i}.beta *this.O{inInd(2)};
                    otherwise
                        assert(false,'Unknown Layer type')
                end
            end
        end
        
        % backward function
        function backward(this,lam)
            
            this.dtheta = this.theta-this.theta;
            
            for i=this.nLayers:-1:1
                outInd = this.net{i}.outInd;
                inInd = this.net{i}.inInd;
                
                if isequal(this.net{i}.type,'affine')
                    this.dtheta(this.net{i}.Ws:this.net{i}.We) = reshape( this.delta{outInd} * this.O{inInd}' , this.net{i}.We-this.net{i}.Ws+1 , 1); % Nabla_W = delta*O{i-1}';
                    this.dtheta(this.net{i}.bs:this.net{i}.be) = reshape( sum(this.delta{outInd},2)' , this.net{i}.be-this.net{i}.bs+1 , 1); % Nabla_b = sum(delta');
                end
                
                if ~(this.net{i}.needBackward)
                    continue;
                end
                
                switch this.net{i}.type
                    case 'loss'
                        this.delta{inInd(1)} = this.net{i}.loss.Grad(this.O(inInd));
                    case 'duplicate'
                        this.delta{inInd} = this.delta{outInd(1)};
                        for j=2:length(outInd)
                            this.delta{inInd} = this.delta{inInd} + this.delta{outInd(j)};
                        end
                    case 'im2col'
                        % method I
                        tmp = cumsum(this.delta{outInd}(this.net{i}.sortedInd));
                        this.delta{inInd} = reshape([tmp(1) ; diff(tmp(this.net{i}.I))] , size(this.O{inInd}));
                    case 'max'
                        s = [size(this.O{inInd},1) 1];
                        tmp = (repmat(this.O{outInd},s) == this.O{inInd});
                        this.delta{inInd} = repmat(this.delta{outInd} ./ sum(tmp),s) .* tmp;
                    case 'mean'
                        s = [size(this.O{inInd},1) 1];
                        this.delta{inInd} = repmat(this.delta{outInd} / s(1),s);
                    case 'reshape'
                        this.delta{inInd} = reshape(this.delta{outInd}, this.net{i}.oldshape);
                        this.O{inInd} = reshape(this.O{outInd}, this.net{i}.oldshape);
                    case 'permute'
                        this.delta{inInd} = permute(this.delta{outInd}, this.net{i}.oldshape);
                    case 'affine'
                        this.delta{inInd} = reshape(this.theta(this.net{i}.Ws:this.net{i}.We),this.net{i}.Wshape)' * this.delta{outInd};
                    case 'relu'
                        this.delta{inInd} = this.delta{outInd} .* (this.O{outInd} > 0);
                    case 'clamp'
                        this.delta{inInd} = this.delta{outInd} .* ((this.O{outInd} > 0) & (this.O{outInd} < 0));
                    case 'concat'
                        tmp = cell(length(size(this.O{outInd})),1);
                        for j=1:length(tmp), tmp{j} = ':'; end;
                        pos=0; nl = size(this.O{inInd(1)},this.net{i}.dim);
                        for j=1:length(inInd)
                            tmp{this.net{i}.dim} = pos + (1:nl);
                            pos = pos+nl;
                            this.delta{inInd(j)} = this.delta{outInd}(tmp{:});
                        end
                    case 'split'
                        this.delta{inInd} = cat(this.net{i}.dim,this.delta{outInd});
                    case 'pad'
                        blobSize = size(this.O{inInd});
                        this.delta{inInd} = this.delta{outInd}(1:blobSize(1),1:blobSize(2),:,:);
                    case 'elementwiseProd'
                        this.delta{inInd(1)} = this.delta{outInd} .* this.O{inInd(2)};
                        this.delta{inInd(2)} = this.delta{outInd} .* this.O{inInd(1)};
                    case 'add'
                        this.delta{inInd(1)} = this.net{i}.alpha * this.delta{outInd};
                        this.delta{inInd(2)} = this.net{i}.beta  * this.delta{outInd};
                    otherwise
                        assert(false,'Unknown Layer type')
                end
                
            end
            
            % and add the regularization gradient
            this.dtheta = this.dtheta + lam*this.theta;
            
        end
        
        
        function Nesterov(this,T,learningRate,mu,lam,param)
            % SGD with Nesterov's momentum
            % See demoMNIST.m for usage
            
            this.prepareForStatAndSnapshot(T,param);
            m = this.net{1}.data.m;
            
            history = this.theta-this.theta;
            
            for t=1:T
                % momentum
                history = mu*history;
                this.theta = this.theta + history;
                
                % choose mini batch
                i = this.net{1}.data.nextRand();
                % forward backward
                this.forward(i);
                this.backward(lam);
                
                % update
                eta = learningRate(t);
                this.theta = this.theta - eta*this.dtheta;
                history = history - eta*this.dtheta;
                
                % statistics for printing and snapshot
                this.statAndSnapshot(t);
            end
        end
        
        function SGD(this,T,learningRate,lam,param)
            % vanilla SGD
            this.prepareForStatAndSnapshot(T,param);
            m = this.net{1}.data.m;

            for t=1:T
                % choose mini batch
                i = this.net{1}.data.nextRand();
                % forward backward
                this.forward(i);
                this.backward(lam);
                
                % update
                eta = learningRate(t);
                this.theta = this.theta - eta*this.dtheta;
                
                % statistics for printing and snapshot
                this.statAndSnapshot(t);
                
            end
        end
        
        
        function SDCA(this,alpha,T,eta,lam,param)
            % SDCA solver
            
            this.prepareForStatAndSnapshot(T,param);
            m = this.net{1}.data.m;
            if ~isempty(alpha)
                % initialize primal from dual
                [d,n] = size(alpha);
                assert(n == m);
                assert(d == length(this.theta));
            else
                % initialize by random dual variabls
                d = length(this.theta);
                n = m;
                alpha = randn(d,n,'single')*lam;
            end
            this.theta = this.mygpuArray(single(mean(alpha,2)/lam));
            beta = eta*lam*n;
            
            for t=1:T
                % choose mini batch
                i = this.net{1}.data.nextRand();
                galpha = this.mygpuArray(alpha(:,i));
                % forward backward
                this.forward(i);
                this.backward(0);
                
                % update
                v = this.dtheta+galpha;
                galpha = galpha - beta*v;
                this.theta = this.theta - eta*v;
                alpha(:,i) = this.mygather(galpha);
                
                % statistics for printing and snapshot
                this.statAndSnapshot(t);
            end
        end
        
        function calcLossAndErr(this)
            m = this.net{1}.data.m;
            this.Loss = this.O{this.lossInd}-this.O{this.lossInd};
            for i=1:m
                this.forward(i);
                this.Loss = this.Loss + this.O{this.lossInd};
            end
            this.Loss = this.Loss/m;
        end
        
        function setTheta(this,newtheta)
            this.theta = this.mygpuArray(newtheta);
        end
        
        
    end
    
    
    % Private methods for initializing the network
    methods (Access = private)
        function initializeNet(this,netD,atGPU)
            % construct a network (net,theta) and initialize the network based on a
            % description given in netD
            
            
            % find maximal value of Oind and required number of layers
            maxOind = 0;
            for i=1:length(netD)
                maxOind = max(maxOind,max(netD{i}.outInd));
            end
            lenO = maxOind;
            this.nLayers = length(netD);
            for i=1:length(netD)
                if strcmp(netD{i}.type,'conv')
                    this.nLayers = this.nLayers + 3;
                    lenO = lenO + 2;
                elseif strcmp(netD{i}.type,'maxpool')
                    this.nLayers = this.nLayers + 2;
                    lenO = lenO + 1;
                elseif strcmp(netD{i}.type,'avgpool')
                    this.nLayers = this.nLayers + 2;
                    lenO = lenO + 1;
                end
            end
            
            
            % initialize
            this.net = cell(this.nLayers,1);
            this.O = cell(lenO,1);
            layerInd = 0;
            this.theta = [];
            needBack = false(lenO,1);
            
            for i=1:length(netD)
                
                Oind = netD{i}.outInd;
                
                % determine the needBackward flag
                if ~strcmp(netD{i}.type,'input')
                    inInd = netD{i}.inInd;
                    needBackward = sum(needBack(inInd))>0;
                end
                
                switch netD{i}.type
                    case 'input'
                        this.O{Oind} = zeros(netD{i}.blobSize,'single');
                        layerInd = layerInd+1;
                        this.net{layerInd} = struct('type','input','outInd',Oind,'inInd',0,'needBackward',false,'data',...
                            dataClass(netD{i}.fName,netD{i}.dataType,netD{i}.blobSize,netD{i}.scale,inf,atGPU));
                        
                    case 'duplicate'
                        layerInd = layerInd+1;
                        this.net{layerInd} = struct('type','duplicate','outInd',Oind,'inInd',inInd,'needBackward',needBackward);
                        for j=Oind
                            this.O{j} = this.O{inInd};
                        end
                        needBack(Oind) = needBack(inInd);
                        
                    case 'split'
                        layerInd = layerInd+1;
                        this.net{layerInd} = struct('type','split','outInd',Oind,'inInd',inInd,'dim',netD{i}.dim,'needBackward',needBackward);
                        blobSize = size(this.O{inInd});
                        for j=1:(length(blobSize))
                            cellInd{j} = ':';
                        end
                        curDim=0; jump=blobSize(netD{i}.dim)/length(Oind);
                        for j=Oind
                            cellInd{netD{i}.dim} = curDim+(1:jump);
                            curDim=curDim+jump;
                            this.O{j} = this.O{inInd}(cellInd{:});
                        end
                        needBack(Oind) = needBack(inInd);
                        
                    case 'concat'
                        layerInd = layerInd+1;
                        this.net{layerInd} = netD{i}; this.net{layerInd}.needBackward = needBackward;
                        this.O{Oind} = cat(netD{i}.dim,this.O{inInd});
                        needBack(Oind) = sum(needBack(inInd))>0;
                        
                        
                    case 'conv'
                        
                        originalBlobDimSize = size(this.O{inInd});
                        if length(originalBlobDimSize)<4
                            originalBlobDimSize = [originalBlobDimSize ones(1,4-length(originalBlobDimSize))];
                        end
                        
                        % construct im2col layer
                        maxOind = maxOind + 1;
                        [layer,blobDim,height,width] = this.constructIm2ColLayer(netD{i}.kernelsize,netD{i}.stride,originalBlobDimSize,needBackward,maxOind,inInd,true);
                        layerInd = layerInd+1;
                        this.net{layerInd} = layer;
                        this.O{maxOind} = zeros(blobDim,'single');
                        needBack(maxOind) = needBack(inInd);
                        
                        % then affine layer
                        nOut = netD{i}.nOutChannels;
                        W = zeros(netD{i}.nOutChannels,blobDim(1));
                        Wind = length(this.theta)+(1:length(W(:)));
                        this.theta = [this.theta ; W(:)];
                        b = zeros(nOut,1) + netD{i}.bias_filler;
                        bind = length(this.theta)+(1:length(b));
                        this.theta = [this.theta ; b];
                        layerInd = layerInd+1;
                        maxOind = maxOind + 1;
                        this.net{layerInd} = struct('type','affine','outInd',maxOind,'inInd',maxOind-1,'ones',this.mygpuArray(ones(1,blobDim(2),'single')),'Ws',min(Wind),'We',max(Wind),'Wshape',size(W),'bs',min(bind),'be',max(bind),'needBackward',needBackward);
                        this.O{maxOind} = zeros(size(W,1),blobDim(2),'single');
                        blobDim = [size(W,1) blobDim(2)];
                        needBack(maxOind) = true;
                        
                        
                        % and then reshape and permute layers
                        % currently, the order in memory is
                        %   (channels,height,width,items)
                        % we want it to be
                        %   (height,width,channels,items)
                        channels = netD{i}.nOutChannels;
                        items = originalBlobDimSize(4);
                        layerInd = layerInd+1;
                        this.net{layerInd} = struct('type','reshape','outInd',maxOind,'inInd',maxOind,'newshape',[channels height width items],'oldshape',[channels height*width*items],'needBackward',true);
                        this.O{maxOind} = reshape(this.O{maxOind},[channels height width items]);
                        needBack(maxOind) = true;
                        layerInd = layerInd+1;
                        this.net{layerInd} = struct('type','permute','outInd',Oind,'inInd',maxOind,'newshape',[2 3 1 4],'oldshape',[3 1 2 4],'needBackward',true);
                        this.O{Oind} = permute(this.O{maxOind},[2 3 1 4]);
                        needBack(Oind) = true;
                        
                        
                    case 'flatten'
                        
                        blobDim = size(this.O{inInd});
                        if length(blobDim)<4
                            blobDim = [blobDim ones(1,4-length(blobDim))];
                        end
                        newshape = [prod(blobDim(1:3)) blobDim(4)];
                        layerInd = layerInd+1;
                        this.O{Oind} = zeros(newshape,'single');
                        this.net{layerInd} = struct('type','reshape','outInd',Oind,'inInd',inInd,'newshape',newshape,'oldshape',blobDim,'needBackward',needBackward);
                        needBack(Oind) = needBack(inInd);
                        
                    case 'affine'
                        
                        blobDim = size(this.O{inInd});
                        ncol = blobDim(2);
                        nrows = blobDim(1);
                        W = zeros(netD{i}.nOutChannels,nrows);
                        Wind = length(this.theta)+(1:length(W(:)));
                        this.theta = [this.theta ; W(:)];
                        b = zeros(netD{i}.nOutChannels,1) + netD{i}.bias_filler;
                        bind = length(this.theta)+(1:length(b));
                        this.theta = [this.theta ; b];
                        this.O{Oind} = zeros(size(W,1),ncol,'single');
                        layerInd = layerInd+1;
                        this.net{layerInd} = struct('type','affine','outInd',Oind,'inInd',inInd,'ones',this.mygpuArray(ones(1,ncol,'single')),'Ws',min(Wind),'We',max(Wind),'Wshape',size(W),'bs',min(bind),'be',max(bind),'needBackward',needBackward);
                        
                        needBack(Oind) = true;
                        
                    case {'maxpool','avgpool'}
                        
                        originalBlobDimSize = size(this.O{inInd});
                        if length(originalBlobDimSize)<4
                            originalBlobDimSize = [originalBlobDimSize ones(1,4-length(originalBlobDimSize))];
                        end
                        
                        
                        % construct im2col layer
                        maxOind = maxOind + 1;
                        [layer,blobDim,height,width] = this.constructIm2ColLayer(netD{i}.kernelsize,netD{i}.stride,originalBlobDimSize,needBackward,maxOind,inInd,false);
                        this.O{maxOind} = zeros(blobDim,'single');
                        layerInd = layerInd+1;
                        this.net{layerInd} = layer;
                        needBack(maxOind) = needBack(inInd);
                        
                        
                        % then max layer
                        blobDim = [1 blobDim(2)];
                        layerInd = layerInd+1;
                        if strcmp(netD{i}.type,'maxpool')
                            this.net{layerInd} = struct('type','max','outInd',Oind,'inInd',maxOind,'needBackward',needBackward);
                        else
                            this.net{layerInd} = struct('type','mean','outInd',Oind,'inInd',maxOind,'needBackward',needBackward);
                        end
                        
                        % and then reshape
                        channels = originalBlobDimSize(3);
                        items = originalBlobDimSize(4);
                        layerInd = layerInd+1;
                        this.net{layerInd} = struct('type','reshape','outInd',Oind,'inInd',Oind,'newshape',[height width channels items],'oldshape',blobDim,'needBackward',needBackward);
                        this.O{Oind} = zeros([height width channels items],'single');
                        needBack(Oind) = needBack(inInd);
                        
                        
                    case 'relu'
                        layerInd = layerInd+1;
                        this.O{Oind} = this.O{inInd};
                        this.net{layerInd} = struct('type','relu','outInd',Oind,'inInd',inInd,'needBackward',needBackward);
                        needBack(Oind) = needBack(inInd);
                        
                    case 'clamp'
                        layerInd = layerInd+1;
                        this.O{Oind} = this.O{inInd};
                        this.net{layerInd} = struct('type','clamp','outInd',Oind,'inInd',inInd,'needBackward',needBackward);
                        needBack(Oind) = needBack(inInd);
                        
                    case 'reshape'
                        layerInd = layerInd+1;
                        this.O{Oind} = reshape(this.O{inInd},netD{i}.newshape);
                        this.net{layerInd} = struct('type','reshape','outInd',Oind,'inInd',inInd,'newshape',netD{i}.newshape,'oldshape',size(this.O{inInd}),'needBackward',needBackward);
                        needBack(Oind) = needBack(inInd);
                        
                    case 'permute'
                        [~,ind] = sort(netD{i}.newshape);
                        layerInd = layerInd+1;
                        this.O{Oind} = permute(this.O{inInd},netD{i}.newshape);
                        this.net{layerInd} = struct('type','permute','outInd',Oind,'inInd',inInd,'newshape',netD{i}.newshape,'oldshape',ind,'needBackward',needBackward);
                        needBack(Oind) = needBack(inInd);
                        
                    case 'pad'
                        layerInd = layerInd+1;
                        blobSize = size(this.O{inInd});
                        this.O{Oind} = zeros(blobSize(1)+netD{i}.amount,blobSize(2)+netD{i}.amount,blobSize(3),blobSize(4),'single');
                        this.net{layerInd} = struct('type','pad','outInd',Oind,'inInd',inInd,'amount',netD{i}.amount,'needBackward',needBackward);
                        needBack(Oind) = needBack(inInd);
                        
                    case 'elementwiseProd'
                        layerInd = layerInd+1;
                        this.O{Oind} = this.O{inInd(1)};
                        this.net{layerInd} = struct('type','elementwiseProd','outInd',Oind,'inInd',inInd,'needBackward',needBackward);
                        needBack(Oind) = sum(needBack(inInd))>0;
                        
                    case 'add'
                        layerInd = layerInd+1;
                        this.O{Oind} = this.O{inInd(1)};
                        this.net{layerInd} = struct('type','add','outInd',Oind,'inInd',inInd,'alpha',netD{i}.alpha,'beta',netD{i}.beta,'needBackward',needBackward);
                        needBack(Oind) = sum(needBack(inInd))>0;
                        
                    case 'loss'
                        layerInd = layerInd+1;
                        this.net{layerInd} = struct('type','loss','outInd',Oind,'inInd',inInd,'loss',lossClass(netD{i}.lossType),'needBackward',needBackward);
                        this.O{Oind} = this.net{layerInd}.loss.LossAndErr(this.O(inInd));
                        needBack(Oind) = needBack(inInd(1));
                        this.lossInd = Oind;
                        
                    otherwise
                        assert(false,'Unknown Layer type')
                end
            end
                        
        end
        
        
        function [layer,blobDim,outHeight,outWidth] = constructIm2ColLayer(this,ksize,kstride,blobDim,needBackward,Oind,inInd,isConv)
            
            B = reshape((1:(prod(blobDim))),blobDim);
            C = [];
            
            if isConv
                for t=1:blobDim(4)
                    w=0;
                    while (w+ksize <= blobDim(2))
                        h=0;
                        while (h+ksize <= blobDim(1))
                            C = [C reshape(B(h+(1:ksize),w+(1:ksize),:,t),ksize*ksize*blobDim(3),1)];
                            h = h+kstride;
                        end
                        w=w+kstride;
                    end
                end
                outHeight =  h/kstride;
                outWidth = w/kstride;
            else % for pooling layer
                outHeight = ceil((blobDim(1) - ksize)/kstride) + 1;
                outWidth = ceil((blobDim(2) - ksize)/kstride) + 1;
                for t=1:blobDim(4)
                    for c=1:blobDim(3)
                        for ww=1:outWidth
                            ws = (ww-1)*kstride + 1;
                            we = min(size(B,2),ws-1+ksize);
                            Iw = zeros(ksize,1) + we;
                            Iw((ws:we)-ws+1) = (ws:we);
                            for hh=1:outHeight
                                hs = (hh-1)*kstride + 1;
                                he = min(size(B,1),hs-1+ksize);
                                Ih = zeros(ksize,1) + he;
                                Ih((hs:he)-hs+1) = (hs:he);
                                C = [C reshape(B(Ih,Iw,c,t),ksize*ksize,1)];
                            end
                        end
                    end
                end
            end
            [val,ind] = sort(C(:));
            I = [find(val(1:end-1) ~= val(2:end)) ; length(val)];
            
            % method II -- not implmenented properly
            % backwardMat = zeros(ksize*ksize,prod(blobDim));
            % backwardMat(1,1) = 1;
            % for j=2:length(I)
            %     backwardMat(1:length(ind((I(j-1)+1):I(j))),j) = ind((I(j-1)+1):I(j));
            % end
            % J = find(backwardMat>0); bI = backwardMat(J);
            %
            % For the above method we will need:
            %this.net{i}.backwardMat(this.net{i}.J) = delta(this.net{i}.bI);
            %delta = reshape( sum(this.net{i}.backwardMat) , size(this.O{this.net{i}.inInd}));
            
            blobDim = size(C);
            layer = struct('type','im2col','outInd',Oind,'inInd',inInd,'im2colInd',this.mygpuArray(uint32(C)),'sortedInd',this.mygpuArray(uint32(ind)),'I',this.mygpuArray(uint32(I)),'needBackward',needBackward);
            %layer = struct('type','im2col','im2colInd',mygpuArray(uint32(C)),'backwardMat',mygpuArray(zeros(size(backwardMat),'single')),'J',mygpuArray(uint32(J)),'bI',mygpuArray(uint32(bI)),'sortedInd',mygpuArray(uint32(ind)),'I',mygpuArray(uint32(I)),'inInd',int32(inInd),'outInd',int32(outInd),'needBackward',needBackward);
            
        end
        
        function newNetD=add_reduced_conv_layers(this,netD)
            
            layerInd=0; maxOind = netD{end}.outInd;
            for i=1:length(netD)
                layerInd = layerInd+1;
                if strcmp(netD{i}.type,'reducedConv')
                    inInd = netD{i}.inInd; outInd = netD{i}.outInd; nC = netD{i}.nOutChannels;
                    % construct conv 1x1
                    maxOind = maxOind + 1;
                    newNetD{layerInd} = struct('type','conv','inInd',inInd,'outInd',maxOind,'kernelsize',1,'stride',1,'nOutChannels',netD{i}.rank*nC,'bias_filler',netD{i}.bias_filler);
                    
                    % construct the split layer
                    layerInd = layerInd+1;
                    tmpInChannels = maxOind+(1:nC);
                    newNetD{layerInd} = struct('type','split','inInd',maxOind,'outInd',tmpInChannels,'dim',3);
                    maxOind = maxOind + nC;
                    
                    % construct the group conv layers
                    tmpOutChannels = tmpInChannels(end) + (1:nC);
                    for j=1:nC
                        layerInd = layerInd+1;
                        newNetD{layerInd} = struct('type','conv','inInd',tmpInChannels(j),'outInd',tmpOutChannels(j),'kernelsize',netD{i}.kernelsize,'stride',netD{i}.stride,'nOutChannels',1,'bias_filler',netD{i}.bias_filler);
                    end
                    
                    % concat layer
                    layerInd = layerInd+1;
                    newNetD{layerInd} = struct('type','concat','inInd',tmpOutChannels,'outInd',outInd,'dim',3);
                    
                else
                    newNetD{layerInd} = netD{i};
                end
            end
            
        end
        
        function prepareForStatAndSnapshot(this,T,param)
            if isfield(param,'printDecay')
                this.printDecay=param.printDecay;
            else
                this.printDecay=0.9;
            end
            if isfield(param,'printIter')
                this.printIter=param.printIter;
            else
                this.printIter=T; this.printDecay=0;
            end
            if isfield(param,'snapshotFile')
                this.snapshotFile = param.snapshotFile;
            else
                this.snapshotFile=[];
            end
            this.Loss=this.O{this.lossInd}-this.O{this.lossInd};
            this.AllLoss=repmat(this.Loss,1,floor(T/this.printIter));
        end
        
        function statAndSnapshot(this,t)
            this.Loss = this.printDecay * this.Loss + (1-this.printDecay)*this.O{this.lossInd};
            if (rem(t,this.printIter)==0)
                this.AllLoss(:,t/this.printIter) = this.Loss;
                fprintf(1,'Iter: %d: ',t); for i=1:length(this.Loss), fprintf(1,'%f ',this.Loss(i)); end; fprintf(1,'\n');
                if ~isempty(this.snapshotFile)
                    fid = fopen(sprintf('%s.%.7d.bin',this.snapshotFile,t),'wb');
                    fwrite(fid,this.mygather(this.theta),'single');
                    fclose(fid);
                end
            end
        end

        
        function initializeWeights(this,initMethod)
            for i=1:length(this.net)
                if strcmp(this.net{i}.type,'affine')
                    switch initMethod
                        case 'Orthogonal'
                            [Q,~] = qr(randn(this.net{i}.Wshape)');
                            W = Q(1:this.net{i}.Wshape(1),:);
                        case 'Xavier'
                            W = (rand(this.net{i}.Wshape)-0.5)/ sqrt(this.net{i}.Wshape(2)) * sqrt(3);
                        otherwise
                            fprintf(1,'Unknown initalization method %s\n',initMethod);
                    end
                    this.theta(this.net{i}.Ws:this.net{i}.We) = W(:);
                end
            end
        end
        
    end
    
end
