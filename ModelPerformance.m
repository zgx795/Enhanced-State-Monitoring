classdef ModelPerformance < handle
    properties
        bestValLoss = inf;
    end

    methods
        function update(this, info)
            if info.ValidationLoss < this.bestValLoss
                this.bestValLoss = info.ValidationLoss;
               % this.bestValLoss = info.State;
              
            end
        end
    end
end

