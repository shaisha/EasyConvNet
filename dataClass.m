classdef dataClass < handle
   
    properties (SetAccess = private)
        X;
        scale;
        atGPU;
        m;
        blobSize;
        cellInd;
    end
    
    
    methods
        function this = dataClass(fName,type,blobSize,scale,maxExamples,atGPU)

            this.atGPU = atGPU;
            this.scale = scale;
            this.blobSize = blobSize;
            
            fid = fopen(fName,'rb');
            this.X = fread(fid,maxExamples*prod(this.blobSize),type);
            this.m = floor(length(this.X)/(prod(this.blobSize)));
            this.X = reshape(this.X(1:(this.m*prod(this.blobSize))),[this.blobSize , this.m]);
            fclose(fid);
            
            this.cellInd = cell(length(size(this.X)),1); 
            for i=1:(length(this.cellInd)-1)
                this.cellInd{i} = ':';
            end
        end

        function x = get(this,i)
            this.cellInd{end} = i;
            if this.atGPU
                x = gpuArray(single(this.X(this.cellInd{:})*this.scale));
            else
                x = single(this.X(this.cellInd{:})*this.scale);
            end
        end
        
    end
end
