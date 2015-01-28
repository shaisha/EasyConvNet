classdef dataClass < handle
   
    properties (SetAccess = private)
        X;
        scale;
        atGPU;
        type;
        m;
        blobSize;
        blobSizeBytes;
        cellInd;
        fid;
        curInd;
        nExamplesInBlock;
        curBlock;
        blockInds;
        nBlocks;
        blockStart;
        blockEnd;
    end
    
    
    methods
        function this = dataClass(fName,type,blobSize,scale,maxExamples,atGPU)

            this.atGPU = atGPU;
            this.scale = scale;
            this.blobSize = blobSize;
            this.type = type;
            
            this.fid = fopen(fName,'rb');
            fread(this.fid,prod(this.blobSize),type);
            this.blobSizeBytes = ftell(this.fid);
            fseek(this.fid,0,'eof');
            lenInBytes = ftell(this.fid);
            this.m = min(floor(lenInBytes/this.blobSizeBytes),maxExamples);
            
            this.nExamplesInBlock = min(this.m, floor(2^30 / this.blobSizeBytes));
            this.nBlocks = ceil(this.m / this.nExamplesInBlock);
            this.curBlock = 0;
            this.fetchBlock(1);
            
            this.cellInd = cell(length(size(this.X)),1); 
            for i=1:(length(this.cellInd)-1)
                this.cellInd{i} = ':';
            end
        end

        function delete(this)
            fclose(this.fid);
        end
        
        function x = get(this,i)
            if i<this.blockStart || i>this.blockEnd
                this.fetchBlock(ceil(i/this.nExamplesInBlock));
            end
            i = mod(i,this.nExamplesInBlock);
            if i==0
                i=length(this.blockInds);
            end
            this.cellInd{end} = i;
            if this.atGPU
                x = gpuArray(single(this.X(this.cellInd{:})*this.scale));
            else
                x = single(this.X(this.cellInd{:})*this.scale);
            end
        end
        
        function ind=nextRand(this)
            if this.curInd == length(this.blockInds)
                this.fetchBlock(randi(this.nBlocks,1,1));
            end
            this.curInd = this.curInd+1;
            ind = this.blockInds(this.curInd);
        end
        
        function fetchBlock(this,ind)
            if ind ~= this.curBlock
                this.curBlock = ind;
                fseek(this.fid,(this.curBlock-1)*this.nExamplesInBlock*this.blobSizeBytes,'bof');
                this.blockStart = (this.curBlock-1)*this.nExamplesInBlock + 1;
                this.blockEnd = min(this.m,this.blockStart - 1 + this.nExamplesInBlock);
                n = this.blockEnd - this.blockStart + 1;
                this.X = reshape(fread(this.fid,n*prod(this.blobSize),this.type),[this.blobSize n]);
            end
            this.blockInds = this.blockStart-1 + randperm(this.blockEnd - this.blockStart + 1);
            this.curInd = 0;
        end
        
    end
end
