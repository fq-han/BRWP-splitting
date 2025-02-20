clear all; close all;
addpath(genpath('utils'));
seed = 10;  
rng(seed);

d = 20; % dimension
beta = 1;
h = 0.1; %stepsize in time
nx = 100; %number of particles
Max_it = 100;
lambdak = 0.1; %L1 regularization parameter
example_idx = 2;

f = @(xk)lambdak.*sum(abs(xk),2);
[g,dg,xex,prhos1,prhos2] = Mixture_example(example_idx,d,lambdak);
 
xinit = 5.*randn(nx,d);  
xk1 = xinit; xk2 = xinit; xk3 = xinit; xk4 = xinit;
err = zeros(6,Max_it);

dg1 = zeros(nx,d); dg3 = dg1;  dg2 = dg1; 
for k = 1:Max_it
    if k == floor(Max_it/2)
        h = h/4;
    end
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
    
    err(1,k)  = TVL1_dist_norm(xk1,bandwidth,1,1,nx,prhos1); 
    err(2,k) = TVL1_dist_norm(xk2,bandwidth,1,1,nx,prhos1); 
    err(3,k) = TVL1_dist_norm(xk3,bandwidth,1,1,nx,prhos1); 
    err(4,k) = TVL1_dist_norm(xk1,bandwidth,1,1,nx,prhos1); 
    err(5,k)  = TVL1_dist_norm(xk2,bandwidth,d,1,nx,prhos2); 
    err(6,k) = TVL1_dist_norm(xk3,bandwidth,d,1,nx,prhos2); 

    fprintf(['iteration ',num2str(k), '\n']);
end
 
hx = 0.01; xx = -20:hx:20;    
yy = prhos1; yy2 = prhos2;
bdL = -10; bdR = 10;
close all;
figure(1);
yy = yy./sum(yy.*hx); yy2 = yy2./sum(yy2.*hx);   
subplot(2,3,1); hold on; plot(xx,yy); histogram(xk1(:,1),floor(size(xk1,1)/16),'Normalization','pdf');   
xlim([bdL, bdR]); hold off; 
subplot(2,3,2); hold on; plot(xx,yy); histogram(xk2(:,1),floor(size(xk2,1)/16),'Normalization','pdf');   
xlim([bdL, bdR]);  hold off; 
subplot(2,3,3); hold on; plot(xx,yy); histogram(xk3(:,1),floor(size(xk3,1)/16),'Normalization','pdf');   
xlim([bdL, bdR]);  hold off; 
subplot(2,3,4); hold on; plot(xx,yy2); histogram(xk1(:,end),floor(size(xk3,1)/16),'Normalization','pdf');   
xlim([bdL, bdR]); hold off; 
subplot(2,3,5); hold on; plot(xx,yy2); histogram(xk2(:,end),floor(size(xk3,1)/16),'Normalization','pdf');   
xlim([bdL, bdR]);  hold off; 
subplot(2,3,6); hold on; plot(xx,yy2); histogram(xk3(:,end),floor(size(xk3,1)/16),'Normalization','pdf');   
xlim([bdL, bdR]);  hold off; 
 
figure(2); 
subplot(2,3,1); scatter(xk1(:,1),xk1(:,2)); subplot(2,3,2); scatter(xk2(:,1),xk2(:,2)); 
subplot(2,3,3); scatter(xk3(:,1),xk3(:,2));  
subplot(2,3,4); scatter(xk1(:,d-1),xk1(:,d)); subplot(2,3,5); scatter(xk2(:,d-1),xk2(:,d)); 
subplot(2,3,6); scatter(xk3(:,d-1),xk3(:,d));  

figure(3);
bandwidth = 0.5;
yp = 0;
for jx = 1:nx
yp = yp + exp(-((xk1(jx,1)-xx).^2)./(2*bandwidth));
end
yp = yp./sum(abs(yp.*hx));
subplot(2,3,1); hold on; plot(xx,yy); plot(xx,yp);
yp = 0;
for jx = 1:nx
yp = yp + exp(-((xk2(jx,1)-xx).^2)./(2*bandwidth));
end
yp = yp./sum(abs(yp.*hx));
subplot(2,3,2); hold on; plot(xx,yy);plot(xx,yp);
yp = 0;
for jx = 1:nx
yp = yp + exp(-((xk3(jx,1)-xx).^2)./(2*bandwidth));
end
yp = yp./sum(abs(yp.*hx));
subplot(2,3,3); hold on; plot(xx,yy); plot(xx,yp);
bandwidth = 2*h;
yp = 0;
for jx = 1:nx
yp = yp + exp(-((xk1(jx,d)-xx).^2)./(2*bandwidth));
end
yp = yp./sum(abs(yp.*hx));
subplot(2,3,4); hold on; plot(xx,yy2); plot(xx,yp);
yp = 0;
for jx = 1:nx
yp = yp + exp(-((xk2(jx,d)-xx).^2)./(2*bandwidth));
end
yp = yp./sum(abs(yp.*hx));
subplot(2,3,5); hold on; plot(xx,yy2);plot(xx,yp);
yp = 0;
for jx = 1:nx
yp = yp + exp(-((xk3(jx,d)-xx).^2)./(2*bandwidth));
end
yp = yp./sum(abs(yp.*hx));
subplot(2,3,6); hold on; plot(xx,yy2); plot(xx,yp);
 
bandwidth = 0.5;
figure(4); bdL = -15; bdR = 15;
subplot(1,4,1); hold on; surf_KDE(xk1,2*bandwidth,bdL,bdR);
subplot(1,4,2); hold on; surf_KDE(xk2,2*bandwidth,bdL,bdR);
subplot(1,4,3); hold on; surf_KDE(xk3,2*bandwidth,bdL,bdR);
xx = linspace(bdL,bdR,200); [x1,x2] = meshgrid(xx,xx); z = zeros(size(x1));
for jx = 1:length(x1(:))
    z(jx) = exp(-f([x1(jx),x2(jx),xex(3:d)])-g([x1(jx),x2(jx),xex(3:d)]));
end
subplot(1,4,4); hold on;   surf(x1,x2,reshape(z,size(x1)),'EdgeColor', 'none', 'FaceAlpha', 1, 'FaceColor','interp'); view(45,20); 

figure(5);
subplot(1,2,1); plot(1:k,err(1,:),'b',1:k,err(2,:),'r',1:k,err(3,:),'k'); legend('xk1','xk2','xk3');
subplot(1,2,2); plot(1:k,err(4,:),'b',1:k,err(5,:),'r',1:k,err(6,:),'k'); legend('xk1','xk2','xk3');

function y = dgs_aux(x,dg)
    y = zeros(size(x));
    for jd = 1:size(x,2)
        y(:,jd) = dg{jd}(x);
    end
end