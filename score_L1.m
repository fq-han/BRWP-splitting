function score = score_L1(xk,lambdak,h,beta)
%score function of lambdak*||x_k||_1
    [nx,d] = size(xk);      
    Sk = sign(xk).*max(abs(xk)-lambdak*h,0); 
    Nxi = exp(beta./2.*(lambdak.*abs(Sk)+(Sk-xk).^2./(2*h)))/nx;
    
    score = zeros(nx,d);
    for jx = 1:nx %parallel when nx > 1000
        rhoxj = 0; drhoxj = 0;   
        expxkjl = exp(-beta/2.*((xk(jx,:) - xk).^2./(2*h)));
        for lx = 1:nx
            rhoxjl = expxkjl(lx,:).*Nxi(lx,:);
            rhoxj = rhoxj +  rhoxjl;   
            drhoxj = drhoxj + xk(lx,:).*rhoxjl;   
        end
       score(jx,:) = drhoxj./rhoxj;
    end
end