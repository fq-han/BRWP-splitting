function score = score_L2(xk,F,iFtFh,z,h,beta)
%score function for ||z-Fxk||_2^2
    [nx,d] = size(xk);      
    T = h; z = z*ones(1,nx);  
    Sk = iFtFh*(xk'+h*F'*z);  
    Nxi = exp(beta./2.*((z-F*Sk).^2+(Sk-xk.').^2./(2*T))).';
    
    score = zeros(nx,d);
    for jx = 1:nx %parallel when nx > 1000
        rhoxj = 0; drhoxj = 0;   
        expxkjl = exp(-beta/2.*((xk(jx,:) - xk).^2./(2*T)));
        for lx = 1:nx
            rhoxjl = expxkjl(lx,:).*Nxi(lx,:);
            rhoxj = rhoxj +  (rhoxjl);   
            drhoxj = drhoxj + xk(lx,:).*rhoxjl;   
        end
       score(jx,:) = drhoxj./rhoxj;
    end
end