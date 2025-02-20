clear all; close all;
addpath(genpath('utils'));
seed = 10;  
rng(seed);

d = 50; % dimension
beta = 1;
h = 0.05; %stepsize in time
nx = 100; %number of particles
Max_it = 100;
example_idx = 6;
lambdak = 3*d/(pi^2)/2*2;
errds = zeros(36,Max_it);

f = @(xk)lambdak.*sum(abs(xk),2);
[g,dg,xex] = Logistic_example(d);

xinit = 5.*randn(nx,d);  
xk1 = xinit; xk2 = xinit; xk3 = xinit; xk4 = xinit;
err = zeros(9,Max_it);

score1 = randn(size(xk1))/h*sqrt(2*h);  
dg1 = zeros(nx,d); dg3 = dg1;  dg2 = dg1; 
for k = 1:Max_it
    % if k == floor(Max_it/2)
    %     h = h/4;
    % end
    for jd = 1:d
        dg1(:,jd) = dg{jd}(xk1);
        dg2(:,jd) = dg{jd}(xk2);
        dg3(:,jd) = dg{jd}(xk3);
    end

    % BRWP_splitting
    xk1 = xk1 - h*dg1;
    score1 = score_L1(xk1,lambdak,h,beta);
    Sk = sign(xk1).*max(abs(xk1)-lambdak*h,0);
    xk1 =  xk1 + 1/2*(Sk - score1);

    % MYULA
    xk2 = sign(xk2).*max(abs(xk2)-lambdak*h,0) - h.*dg2 + 1.*sqrt(2*h)*randn(size(xk2));    
    
    % RGO 
    M = 1; %smoothness constant
    hrgo = h; %theory says h < 1/(sqrt(M*d)) 
    fg_den = @(x)g(x) + lambdak*sum(abs(x));
    dfs = @(x)dgs_aux(x,dg);
    [xk3,rej] = RGO_L1(fg_den,dfs,xk3,hrgo,M,d,1,1,1,lambdak);
 
    bandwidth = 0.1;
    
    %% error in estimating the mean
    err(1,k) = norm(mean(xk1)-reshape(xex,[1,d]),1)./d;
    err(2,k) = norm(mean(xk2)-reshape(xex,[1,d]),1)./d;
    err(3,k) = norm(mean(xk3)-reshape(xex,[1,d]),1)./d;

    fprintf(['iteration ',num2str(k), '\n']);
end
 
plot(1:k,log(err(1,:)),'b',1:k,log(err(2,:)),'r',1:k,log(err(3,:)),'k');
 
function y = dgs_aux(x,dg)
    y = zeros(size(x));
    for jd = 1:size(x,2)
        y(:,jd) = dg{jd}(x);
    end
end

function [g,dg,xex] = Logistic_example(d)
    dg = cell(1,d);
    %% Logistic regression xk: nx d, Y: nn 1, X: nn d 
    nn =d*10;  alpha = 3*d/(pi^2)/2; %% nn is the number of data, d is parameter dimension
    xex = zeros(1,d); xex(1:floor(d/4)) = 1; 
    X = 2.*binornd(1,0.5,nn,d)-1; 
    Y = zeros(nn,1);  
    SigmaX = 1/nn.*(X'*X);
    for jn = 1:nn
        Y(jn) = binornd(1,exp(xex*X(jn,:)')/(1+exp(xex*X(jn,:)')));
    end
    g = @(xk)-xk*X'*Y + sum(log(1+exp(xk*X')),2) + alpha*sum((xk*SigmaX).*xk,2);
    for jd = 1:d
        dg{jd} = @(xk)-X(:,jd)'*Y+sum((X(:,jd)*ones(1,size(xk,1)))'./(1+exp(-xk*X')),2)+alpha*xk*SigmaX(:,jd);
    end  
end