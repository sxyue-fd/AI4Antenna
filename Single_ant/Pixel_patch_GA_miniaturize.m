% This is adapted from MATLAB official example 
% "Miniaturize Rectangular Microstrip Patch Antenna Using Genetic Algorithm Optimization"
%-----------Create a conventional patch antenna fed by ms line------------
% Stepped feed parameters
inputs.f2Width = 8.34e-3;
inputs.f1Width = 10e-3;
inputs.f2Length = 1.5e-3;
inputs.f1Length = 2.87e-3;

% Substrate parameters
inputs.height = 1.6e-3; 
inputs.Subs = dielectric("FR4");
inputs.Subs.Thickness = inputs.height;

% Ground plane parameters
inputs.groundPlaneLength = 25e-3; 
inputs.groundPlaneWidth = 50e-3;

% Create the feed strip
f1 = antenna.Rectangle(Length=inputs.f1Length, Width=inputs.f1Width, Center=[0, -inputs.groundPlaneWidth/2+inputs.f1Width/2]);
f2 = antenna.Rectangle(Length=inputs.f2Length, Width=inputs.f2Width, Center=[0, -inputs.groundPlaneWidth/2+inputs.f1Width+inputs.f2Width/2]);
inputs.feedArea = f1 + f2;

% Feed location
inputs.feedLocation = [0, -inputs.groundPlaneWidth/2, inputs.height];

% Create the ground plane
inputs.ground = antenna.Rectangle(Length=inputs.groundPlaneLength, Width=inputs.groundPlaneWidth, Center=[0 0]);

% Patch parameters
inputs.patchLength = 17.58e-3;
inputs.patchWidth = 13.85e-3;

patchRect5GHz = antenna.Rectangle(Length=inputs.patchLength, Width=inputs.patchWidth, ...
    Center=[0, -inputs.groundPlaneWidth/2+inputs.f1Width+inputs.f2Width+inputs.patchWidth/2]);

finalPatch5GHz  = inputs.feedArea + patchRect5GHz; 

antAt5GHz = pcbStack(Layers={finalPatch5GHz, inputs.Subs inputs.ground}, BoardThickness=inputs.height, ...
    BoardShape=inputs.ground, FeedLocations=[inputs.feedLocation(1:2), 1, 3]);
%----------------------------------------------------------------

% ---------------Plot antenna structure--------------------
% figure
% show(antAt5GHz);
%----------------------------------------------------------------

% -----------------Compute s parameters for conventional patch------------
% figure 
% rfplot(sparameters(antAt5GHz,linspace(4e9,6e9,21)));
%----------------------------------------------------------------

% -------------Define pixel units for Pixelated patch----------------
inputs.rows = 10;
inputs.cols = 10;
inputs.patchLength = 17.58e-3;
inputs.patchWidth = 13.85e-3;
inputs.unitPatches = cell(inputs.rows, inputs.cols);
inputs.unitPatchLength = inputs.patchLength/inputs.cols;
inputs.unitPatchWidth = (inputs.patchWidth/inputs.rows);
inputs.unitPatchCenters = [];
inputs.overlapWidth = 0.2e-3;% lo: Small margin to keep connection intact
inputs.commCenterX = -inputs.patchLength/2;
inputs.commCenterY = -inputs.groundPlaneWidth/2+inputs.f1Width+inputs.f2Width;

for i = 1:inputs.rows
    for j = 1:inputs.cols
        % Define pixels
        inputs.unitPatches{i, j} = antenna.Rectangle(Length=inputs.unitPatchLength+inputs.overlapWidth/2, ...
            Width=inputs.unitPatchWidth+inputs.overlapWidth/2, ...
            Center=[(i-1)*inputs.unitPatchLength+inputs.unitPatchLength/2+inputs.commCenterX (j-1)*inputs.unitPatchWidth+inputs.unitPatchWidth/2+inputs.commCenterY]);
        inputs.unitPatchCenters = [inputs.unitPatchCenters; inputs.unitPatches{i, j}.Center];        
    end
end
%------------------------------------------------------------------------

%------------------Define the cost function for GA-----------------------
function val = objectiveFunction(inputs, designVector)
% objectiveFunction evaluates the main goal of optimization and passes the output back to the optimizer for each iteration.
% Pixelated patch is generated and simulated iteratively.

% Reshape the input from optimizer into a 10 X 10 matrix
designVector = reshape(designVector, 10, 10);
designVector(5:6,1) = 1; % Patch is forced to electrically connect the feed line

patchNew = copy(inputs.feedArea);

% Add pixels in the patch top layer when value is 1
    for i = 1:inputs.rows
        for j = 1:inputs.cols
            if designVector(i, j) == 1 
                patchNew = patchNew + inputs.unitPatches{i, j};
            end
        end
    end

    % Add overlap when [1 0;0 1] or [0 1; 1 0]
    for i = 1:inputs.rows-1
        for j = 1:inputs.cols-1
            dV4 = designVector(i:i+1,j:j+1);
            if all(all(dV4 == [1 0;0 1])) || all(all(dV4 ==[0 1;1 0]))
                overlap = antenna.Rectangle(Length=2*inputs.unitPatchLength,Width=inputs.overlapWidth,...
                    Center=[i*inputs.unitPatchLength+inputs.commCenterX, j*inputs.unitPatchWidth+inputs.commCenterY]);
                patchNew = patchNew + overlap;
            end
        end
    end


    % Create pcbStack
    ant = pcbStack(Layers={patchNew, inputs.Subs inputs.ground}, BoardThickness=inputs.height, ...
        BoardShape=inputs.ground, FeedLocations=[inputs.feedLocation(1:2), 1, 3]);

    % Objective function

    N = 10; % Total sampling points
    fd = 2.16e9; % Desired frequency
    epsilon = 0.1e9; % Upper - Lower margin
    f =  linspace(fd-epsilon, fd+epsilon, N); % Sampling frequency

    s = sparameters(ant, f);
    S11 = rfparam(s,1,1); % S11(dB) parameter at resonance frequency
    cost = 0;

    for i = 1:N
        fres = f(i); % The individual resonance frequency        
        cost = cost + abs(S11(i))*exp(-20*abs((fres-fd)/fd));
    end
    val = cost;

end

%---------------------GA for miniaturization-----------------------------
figure
% Optimization Options
options = optimoptions('ga',UseParallel=true,PlotFcn='gaplotbestf');
[x, fval, exitflag, output, population, scores] = ...
    ga(@(inputMat)objectiveFunction(inputs, inputMat), ...
    inputs.rows*inputs.cols, ... % Number of design variables
    [], [], [], [], ... % No linear constraints input
    zeros(1, inputs.rows*inputs.cols), ... % lower bound
    1.*ones(1, inputs.rows*inputs.cols), ... % upper bound
    [], ... % No nonlinear constraints
    1:inputs.rows*inputs.cols, ... % Integer constraint
    options); % options