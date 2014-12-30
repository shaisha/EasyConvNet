classdef lossClass < handle
   
    properties (SetAccess = private)
        type;
    end
    
    
    methods
        function this = lossClass(lossType)
            
            switch lossType
                case 'MCLogLoss'
                    this.type = 1;
                case 'SquaredLoss'
                    this.type = 2;
                case 'BinLogLoss'
                    this.type = 3;
                otherwise
                    assert(false,'Unknown loss type')
            end
            
        end
        
        function loss=LossAndErr(this,input)
            switch this.type
                case 1 % multiclass logistic loss
                    [k,m] = size(input{1});
                    [loss,ind]=max(input{1});
                    pred = input{1}-repmat(loss,k,1);
                    [~,y]=max(input{2});
                    err = sum(ind~=y)/m;
                    
                    valY = sum(pred.*input{2});
                    loss = pred - repmat(valY,k,1);
                    loss = sum(log(sum(exp(loss))))/m;
                    loss = [loss;err];
                case 2 % Squared Loss
                    loss = 0.5*mean(sum((input{1}-input{2}).^2));
                    
                case 3 % binary log loss
                    loss = -input{1}.*input{2};
                    err = mean(mean(loss>=0));
                    loss(loss>0) = loss(loss>0) + log(1+exp(-loss(loss>0)));
                    loss(loss<=0) = log(1+exp(loss(loss<=0)));
                    loss = [mean(mean(loss)) ; err];
            end
        end

        function delta=Grad(this,input)
            switch this.type

                case 1 % multiclass logistic loss
                    bla = input{1}-repmat(max(input{1}),size(input{1},1),1);
                    bla = exp(bla);
                    bla=bla./repmat(sum(bla),size(bla,1),1);
                    delta = (bla - input{2})/size(bla,2);
                    
                case 2 % SquaredLoss
                    delta = (input{1}-input{2})/size(input{2},2);
                    
                case 3 % binary log loss
                    delta = -input{2}./(1+exp(input{1}.*input{2}))/prod(size(input{2}));
                    
            end
            
        end
        
        
    end
end
