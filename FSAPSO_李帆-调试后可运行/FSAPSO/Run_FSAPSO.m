%********************************************************************
% Script file for the initialization and run of the differential 
% evolution optimizer.
%********************************************************************
clear,clc,tic;
warning('off');
rng('default');
rng('shuffle');

funcnum=1;

if funcnum==1 %---Ellipsoid 
    fname='Ellipsoid';
    Xmin=-5.12;Xmax=5.12;
    VRmin=-5.12;VRmax=5.12;
elseif funcnum==2 %---Rosenbrock
    fname='Rosenbrock';
    Xmin=-2.048;Xmax=2.048;
    VRmin=-2.048;VRmax=2.048;
elseif funcnum==3 %---Ackley 
    fname='Ackley';
    Xmin=-32.768;Xmax=32.768;
    VRmin=-32.768;VRmax=32.768;
elseif funcnum==4 %---Griewank
    fname='Griewank';
    Xmin=-600;Xmax=600;
    VRmin=-600;VRmax=600;
elseif funcnum==5 %---Rastrigins 
    fname='Rastrigin';
    Xmin=-5.12;Xmax=5.12;
    VRmin=-5.12;VRmax=5.12;
elseif funcnum==10 || funcnum==16 || funcnum==19 % CEC 2005 function F10/F16/F19
    fname='benchmark_func';
    Xmin=-5;Xmax=5;
    VRmin=-5;VRmax=5;
end

% I_D		number of parameters of the objective function 
        I_D = 10;% Common benchmark test suit
          
%         I_D = 30;% Energy minimization of a model molecular configuration: N=12.
%         I_D = 54; % 36D & 54D Novel Passive Vibration Isolator Feasibility
%         I_D = 31; % MOON ET AL. (2012) FUNCTION
% FVr_minbound,FVr_maxbound   vector of lower and upper bounds of initial population
%         the algorithm seems to work especially well if [FVr_minbound,FVr_maxbound] 
%         covers the region where the global minimum is expected
%         *** note: these are no bound constraints!! ***
        FVr_minbound = Xmin.*ones(1,I_D); % Common Benchmark Function
        FVr_maxbound = Xmax.*ones(1,I_D); % Common Benchmark Function
        
%         FVr_minbound = [-15,-3];  % sixth Bukin function
%         FVr_maxbound = [-5, 3]; % sixth Bukin function

%         FVr_minbound = [-3,-2];  % six-hump Camel function ***
%         FVr_maxbound = [3,2];   % six-hump Camel function ***
%         
%         FVr_minbound = Xmin; % Energy minimization of a model molecular configuration
%         FVr_maxbound = Xmax; % Energy minimization of a model molecular configuration
        
%         FVr_minbound = repmat([0.01,0.30,0.50],1,I_D/3); % Stepped Cantilever Beam Design Problem
%         FVr_maxbound = repmat([0.05,0.65,1.00],1,I_D/3); % Stepped Cantilever Beam Design Problem
        
%         FVr_minbound = [0.0003, 298,    0.25,   3.5,    81.2];  % A direct methanol fuel cell system
%         FVr_maxbound = [0.08,   343,    2,      5.5,    140.8]; % A direct methanol fuel cell system

if I_D<=30;maxNFE=11*I_D;else;maxNFE=1000;end     

maxIter = 100;

Dat.I_D          = I_D;
Dat.maxIter      = maxIter;
Dat.FVr_minbound = FVr_minbound;
Dat.FVr_maxbound = FVr_maxbound;

%********************************************************************
% Start of optimization
%********************************************************************

p=gcp('nocreate');
if isempty(p)
    parpool(5);%open
% else
%     delete(gcp('nocreate'));%close
end

runs = 30;
time_begin = tic;

parfor r=1:runs
% for r=1:runs
    r
    [fy,fNFE,t1,pd] = f_sapso(fname,Dat);% MADE_LS
    gsamp1(r,:)=fNFE(1:maxNFE);
    population_diversity(r,:)=pd; % for population diversity visualization when terminated ny I_itermax
end
best_samp=min(gsamp1(:,end));
worst_samp=max(gsamp1(:,end));
samp_mean=mean(gsamp1(:,end));
samp_median=median(gsamp1(:,end));
std_samp=std(gsamp1(:,end));
out1=[best_samp,worst_samp,samp_median,samp_mean,std_samp];
gsamp1_ave=mean(gsamp1,1); 
gsamp1_log=log(gsamp1_ave);
sn1 = 1;
for j=1:maxNFE
    if mod(j,sn1)==0
        j1=j/sn1; gener_samp1(j1)=j;
    end
end

figure(1);
plot(gener_samp1,gsamp1_log,'.-k','Markersize',16)
legend('FSAPSO'); 
% xlim([100,I_evalmax]);
xlabel('Function Evaluation Calls');
ylabel('Mean Fitness Value (log)');
% title('2014 CEC Expensive Benchmark Function (F15)')
% title('2005 CEC Benchmark Function (F10)')
% title('Ackley Function')
% title('Griewank Function')
% title('Rastrigin Funtion')
% title('Rosenbrock Function')
% title('Ellipsoid Function')

% title('Stepped Cantilever Beam Design Problem')
% title('A direct mechanol fuel cell')
% title('Novel Passive Vibration Isolator Feasibility')
% title('MOON ET AL. (2012) FUNCTION')
% title('Energy minimization of a model molecular configuration')

% title('Schwefel Function')
% title('Sphere Function')
% title('Michalewicz Function')
% title('Powell Function')
% title('Zakharov Function')
% title('Levy Function')
% title('Rotated Hyper-Ellipsoid Function')
% title('Dixon-Price Function')
% title('Styblinski-Tang Function')
% title('Langermann Function')
set(gca,'FontSize',20);

time_cost=toc(time_begin);
