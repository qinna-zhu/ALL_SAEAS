%% ********************************************************************************************************************************************************8
% function  [fy,fNFE,t1]=f_sapso(fun,D,lb,ub)
% Usage:  [fy,fNFE,t1]=f_sapso(fun,D,lb,ub)
% -----------------------------------------------------------------------------
% -----------------------------------------------------------------------------
% Input:
% fun           - Name of the problem
% D              - Dimension of the problem
% lb             - Lower Boundary of Decision Variables
% ub            - Upper Boundary of Decision Variables
%
% Output: 
% fy             - the obtained best fitness value
% fNFE         -  the obtained best fitness value in each iteration
% t1             - Execution Time
%--------------------------------------------------------------------------------------
% -------------------------------------------------------------------------------------
% Authors:      Fan Li, Weiming Shen, Xiwen Cai, Liang Gao, G. Gary Wang
% Address       Huazhong University of Science & Technology, Wuhan, PR China;    Simon Fraser University, BC, Canada
% EMAIL:        D201780171@hust.edu.cn
% WEBSITE:    https://sites.google.com/site/handingwanghomepage
% DATE:         April 2020
%This code is part of the program that produces the results in the following paper:
%Fan Li, Weiming Shen, Xiwen Cai, Liang Gao, G. Gary Wang, A Fast Surrogate-Assisted Particle Swarm Optimization Algorithm for Computationally Expensive Problems, Applied Soft Computing, 2020.
%You are free to use it for non-commercial purposes. However, we do not offer any forms of guanrantee or warranty associated with the code. We would appreciate your acknowledgement.
%% *********************************************************************************************************************************************************************************************************

function [fy,fNFE,t1,population_diversity]=f_sapso(fun,Dat)
% clc;clear all;close all;lb=-5;ub=5;D=10; fun=@Rastrigin;%RosenbrockEllipsoid

D = Dat.I_D; % add by Haibo Yu
lb = Dat.FVr_minbound; % add by Haibo Yu
ub = Dat.FVr_maxbound; % add by Haibo Yu
maxIter = Dat.maxIter; % add by Haibo Yu

t0=cputime;
Dat.myFN =fun;
% Dat.designspace = [lb;ub]*ones(1,D);   % lower bound % upper bound
Dat.designspace = [lb;ub];   % modified by Haibo Yu
Dat.ndv = D;
bound=Dat.designspace;
vbound(2,:)=0.1*(bound(2,:)-bound(1,:));vbound(1,:)=-vbound(2,:);

if D<=30;maxNFE=11*D;else;maxNFE=1000;end;

%% 初始化样本设计create DOE
k=max(D,20);% 样本数目
Dat.npoints =k;Doe=@DOELHS ;%@DOEOLHS
[Xtrain, Ytrain]=my_initial(Dat,Doe);
NFE=k;


%% 初始化种群数目
[val ind ]=sort(Ytrain);
if D<100;N=20;else;N=40;end
pop=Xtrain(ind(1:N),:);fpop=Ytrain(ind(1:N));pbest=pop;   fpbest=fpop;
gbest=Xtrain(ind(1),:);fgbest=fpbest(1);
Dat.npoints =N;
[u,~]=my_initial(Dat,Doe);
u=0.25*u; v=u;         %0.5*pop初始速度
w=0.729;   c1=1.491;  c2=1.491;wmax=0.729;wmin=.2;

%%
fNFE(1:NFE)=fgbest;Samp=[Xtrain];YS=[Ytrain];

t=1;
dlta=min(sqrt(0.000001^2*D),0.00005*sqrt(Dat.ndv)*min(Dat.designspace(2,:)-Dat.designspace(1,:)));

g1=0;g2=0;g3=0;g4=0;
t=1;tep1=0;
%% 开始迭代
% while NFE< maxNFE
while (t < maxIter)    % for population diversity visualization -- add by Haibo Yu

    tepg=0;tg=0;   
    G(t,:)=gbest;  FG(t)=fgbest;
    srgtOPT=srgtsRBFSetOptions(Samp,YS, @my_rbfbuild, [],'CUB', 0.0002,1);
    srgtSRGT = srgtsRBFFit(srgtOPT);
    L2 =@(x)my_rbfpredict(srgtSRGT.RBF_Model, srgtSRGT.P, x);
   FE=3000;
    options = optimset('Algorithm','interior-point','Display','off','MaxFunEvals',FE,'TolFun',1e-8,'GradObj','off'); % run interior-point algorithm
    L=min(pop);U=max(pop);
    if isnan(L2(gbest))==0
        x= fmincon(L2,gbest,[],[],[],[],L,U,[],options);
        dx=min(sqrt(sum((repmat(x,size(Samp,1),1)-Samp).^2,2)));
        if dx>dlta
            fx=feval(Dat.myFN,x);   
            Samp=[ Samp;x]; YS=[YS;fx];            
            NFE=NFE+1;  fNFE(NFE)=min(YS);tg=1;           
            if fx<fgbest
                fgbest= fx; gbest=x;%PAU=[PAU;x];FPAU=[FPAU;fx];
                tepg=1;g1=g1+1;
            end            
        end
    end
    
    %% 构造引导的粒子
    for i=1:N
        for d=1:D
            v(i,d)=w*v(i,d)+rand*c1*(gbest(d)-pop(i,d))+rand*c2*(pbest(i,d)-pop(i,d));%
        end
        for j=1:D
            if  v(i,j)<vbound(1,j)
                v(i,j)=vbound(1,j);
            end
            if  v(i,j)>vbound(2,j)
                v(i,j)=vbound(2,j);
            end
        end
        pop1(i,:)=pop(i,:)+v(i,:);
        for j=1:D
            if pop1(i,j)<bound(1,j)
                pop1(i,j)=bound(1,j);
            end
            if  pop1(i,j)>bound(2,j)
                pop1(i,j)=bound(2,j);
            end
        end
        newpop(i,:)=pop1(i,:);
        pop(i,:)=pop1(i,:);
    end
    %% 更新个体、全局最优
    if tg==1
        srgtOPT=srgtsRBFSetOptions(Samp,YS, @my_rbfbuild, [], 'CUB',0.0002,1);
        srgtSRGT = srgtsRBFFit(srgtOPT);
    end
    
    predy= my_rbfpredict(srgtSRGT.RBF_Model, srgtSRGT.P, newpop);
    %% 最小值
    [val,ind]=min(predy);
    dx=min(sqrt(sum((repmat(newpop(ind,:),size(Samp,1),1)-Samp).^2,2)));
    if dx>dlta
        fnew=feval(Dat.myFN, newpop(ind,:));
        Samp=[ Samp; newpop(ind,:)]; YS=[YS;fnew];
        NFE=NFE+1; fNFE(NFE)=min(YS);        
        if fnew<fpop(ind)
            pop(ind,:) = newpop(ind,:);fpop(ind)=fnew;
        end
        if fpop(ind)<fpbest(ind)
            pbest(ind,:)= pop(ind,:);fpbest(ind)=fpop(ind);
        end
        if fnew<fgbest
            gbest= newpop(ind,:);fgbest=fnew;tepg=1;g2=g2+1;
        end
    end
    %% 不确定性
    if tepg==0
        g4=g4+1;
        F=dvar(Samp,YS,newpop);
        [val,ind]=sort(F);
        i=ind(1);
        fnewpop(i)=feval(Dat.myFN, newpop(i,:));
        Samp=[ Samp; newpop(i,:)]; YS=[YS;fnewpop(i)];
        NFE=NFE+1; fNFE(NFE)=min(YS);%AU=[AU;newpop(i,:)];FAU=[FAU;fnewpop(i)];        
        if fnewpop(i)<fpop(i)
            pop(i,:) = newpop(i,:);fpop(i)=fnewpop(i);
        end
        if fpop(i)<fpbest(i)
            pbest(i,:)=pop(i,:);fpbest(i)= fpop(i);
        end
        if fpbest(i)<fgbest
            gbest= pbest(i,:);fgbest=fpbest(i);g3=g3+1;
        end
    end
    
    % ------ Calculate the diversity of the generated parent population % add by Haibo Yu
    fit = zeros(N,1);
    for i = 1:N
        fit(i) =feval(Dat.myFN,pop(i,:));
    end
    parent_population = [pop,fit];
    pd = PD(parent_population);
    population_diversity(t) = log(pd);
    %--------------------------------------% add by Haibo Yu
    
    w=wmax-(wmax-wmin)*NFE/maxNFE;%     w=wmax-(wmax-wmin)*t/tmax;
    NFE;
    t=t+1;
end

fy=fgbest;
t1=cputime-t0;