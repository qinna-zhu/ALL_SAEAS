%% ********************************************************************************************************************************************************8
% function   [y dg va]=dvar(Xtrain,Ytrain,Xtest)      fitness value (DF) criterion
% Usage:     [y dg va]=dvar(Xtrain,Ytrain,Xtest)
% -----------------------------------------------------------------------------
% -----------------------------------------------------------------------------
% Input:
%Xtrain              - the input of the sample
%Ytrain              - the output of the sample
% Xtest               - the input of the candidate points
%
% Output: 
% y             - fitness value (DF) criterion

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




function  [y dg va]=dvar(Xtrain,Ytrain,Xtest)
[m,d]=size(Xtest);
n0=3;%1  5  10
for i=1:m
    d=pdist2(Xtest(i,:),Xtrain);
    [val,ind]=sort(d);  dmin(i)=val(1);
    dg(i)=mean(val(1:n0));
    t1=Ytrain(ind(1:n0));
    va(i)=sqrt((var(t1)));
end

dmin=5*dmin/sum(dmin);
s=1./(1+exp(-dmin))-0.5;
dg=dg/sum(dg);
va=va/sum(va);
y=-s.*(dg+va);

