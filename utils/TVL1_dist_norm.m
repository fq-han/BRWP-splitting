function fval = TVL1_dist_norm(xk,bandwidth,dim,d,nx,prhos)
    cnorm = (2*pi*bandwidth)^(-d/2)/nx;
    % prhok = @(y,xk)kde_handel(y,xk,nx,bandwidth,cnorm);
    % fval = 0;  
    % for jx = 1:nsamples
    %     zj = randn(nx,d).*sqrt(bandwidth) + xk;
    %     fval = fval + sum(log(prhok(zj,xk)./prhos(zj)));
    % end
    xx = -20:0.01:20;
    prhox = 0;
    for jx = 1:nx
    prhox =  prhox + exp(-(xx-xk(jx,dim)).^2./(2*bandwidth));
    end
    prhox = prhox.*cnorm;
    % fval = sum(prhox.*log(max(prhox,0.0001)./max(prhos,0.0001))).*0.01;
    fval = sum(abs(prhox-prhos).*0.01);
    % fval = fval/(nsamples*nx); 

end

function prhok = kde_handel(y,xk,nx,bandwidth,cnorm)
    prhok = 0;
    parfor jx = 1:nx
        prhok = prhok + exp(-sum(y-xk(jx,1),2).^2./(2*bandwidth));
    end
    prhok = prhok.*cnorm;
end