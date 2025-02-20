clear all;
addpath(genpath('utils'));
seed = 10;
rng(seed);

h = 0.1; T = h;  
lambdak = 2; coe2 = 1;

%% noise 
f=imread('img_set\pout.jpg');
f = f(:,:,1);
image(f); colormap(gray); axis image; title('input image f'); 
N_img = min(size(f));  NN = 200; bndidx = floor((N_img-NN)/2);
xex =double(f(bndidx:NN+bndidx-1,bndidx:NN+bndidx-1));
xex = xex(:)';
xex = xex./100; %scale for better numerical stability, the kernel formula may be unstable if value of particle is too large due to the exponential function

dx = NN^2; nx = 20; beta = 1;   
dm = 2*dx;

%distribution is ||Ax-g||_2^2 + lambdak ||m||_1 - lambak ||m||_2 + y[I,-grad](m,x)
nz = dx;  
A = speye(nz,dx) + 0.1.*sprandn(nz,dx,4*1e-5);
AA = A'*A;  iAtAh = speye(dx)-h.*AA;
z = A*xex'; 
fd_matrix = fd_mat_2d(NN);
z = z.*(1+0.2.*randn(size(z))); %noisy measurement

xinit = ones(nx,1)*z'+4.*randn(nx,dx);  
xk1 = xinit; xk2 = xinit; 
mk1 = (fd_matrix*(xk1'))';
mk2 = (fd_matrix*(xk2'))';

yk1 = 1.*ones(nx,dm); yk2 = 1.*ones(nx,dm);
Max_it = 20;  

nu = 1; 
err = zeros(Max_it,2);
for k = 1:Max_it
    % if k < 50
    %     h = 0.1; T = h;  Params.h = h;
    % elseif k < 100
    %     h = 0.1;  T = h; Params.h = h;
    % elseif k < 200
    %     h = 0.1;  T = h; Params.h = h;
    % else
    %     h = 0.1;  T = h ;Params.h = h;
    % end
    
    pxk1 = xk1; pmk1 = mk1;
     
    %step 1 gradident descent with y
    hxk1 = xk1 + (fd_matrix'*(yk1'))';
    hmk1 = mk1 - yk1;

    %step 2 gradient descent with x
    Ax1z = A*hxk1'- z*ones(1,nx);  
    nhxk1 = sqrt(sum(hxk1.^2,2))*ones(1,dx);
    AtA = 2*Ax1z'*A*coe2;
    dg1 = AtA;  
    % xk1 = hxk1 - h*dg1;%+ sqrt(2*h).*randn(size(xk1)); 
    scorehxk = score_L2(hxk1,A,iAtAh,z,h,beta);
    xk1 =  hxk1  - lambdak.*hxk1./nhxk1  + 1/2*(hxk1 - h*dg1 -scorehxk);
    %TV L2

    tic;
    Sk = sign(hmk1).*max(abs(hmk1)-lambdak*h,0);
    scorehmk = score_L1(hmk1,lambdak,h,beta);
    mk1 =   hmk1  + 1/2*(Sk - scorehmk);
    toc;

    % step 3 gradient ascent with y and hard thresholoding
    yk1 = yk1 - h*(fd_matrix*(2.*xk1-pxk1)')'+h*(2.*mk1-pmk1);
    yk1 = yk1./max(abs(yk1),1);

    %PDHG + Brwonian motion / MYULA
    tic;
    pxk2 = xk2; pmk2 = mk2;
    hxk2 = xk2 + (fd_matrix'*(yk2'))';  
    hmk2 = mk2 - yk2;
    Ax2z = A*hxk2'- z;  
    nhxk2 = sqrt(sum(hxk2.^2,2))*ones(1,dx);
    AtA = 2*Ax2z'*A*coe2;
    dg2 = AtA - lambdak.*hxk2./nhxk2;
    xk2 = hxk2 - h*dg2 + sqrt(2*h).*randn(size(xk2)); 
    Sk2 = sign(hmk2).*max(abs(hmk2)-lambdak*h,0);
    mk2 =  Sk2 + sqrt(2*h).*randn(size(mk2)); 
    yk2 = yk2 + h*(-(fd_matrix*(2.*xk2-pxk2)')'+(2.*mk2-pmk2));
    yk2 = yk2./max(abs(yk2),1);
    toc;
    fprintf(['iteration ',num2str(k), '\n']);

    err(k,:) = [norm(mean(xk1)-xex),norm(mean(xk2)-xex)];
    if mod(k,10) == 0
        figure(1); 
        subplot(1,4,1); imagesc(reshape(xex,[NN,NN])); colormap('gray');
        subplot(1,4,2); imagesc(reshape(z,[NN,NN])); colormap('gray');
        subplot(1,4,3); imagesc(reshape(mean(xk1),[NN,NN])); 
        subplot(1,4,4); imagesc(reshape(mean(xk2),[NN,NN])); 
    end
end
function fd_matrix = fd_mat_2d(NN)
   grad_mat = diag(ones(NN-1,1),1) + diag(-ones(NN,1),0); 
   grad_mat(NN,NN-1:NN) = [-1,1]; 
   fd_matrix1 = kron(speye(NN),grad_mat);
   fd_matrix2 = kron(grad_mat,speye(NN));
   fd_matrix = [fd_matrix1;fd_matrix2];
   fd_matrix = sparse(fd_matrix);
end
 
 