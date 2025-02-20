function [output,rej] = RGO_L1(fg_den,dfs,xk,h,M,d,n,alpha,delta,lambdak)
    % h step size, alpha, delta, M see parameters in Liang's paper
    yk = xk - h*dfs(xk);
    % prod = @(a,b) sum(dot(a,b));
    output = zeros(size(xk));
    ws = sign(yk).*max(abs(yk)-lambdak*h,0);
    pf = @(x,y)exp(-1/(4*h)*sum((y-(x-h*dfs(x))).^2)-lambdak*sum(abs(y)));
    for jx = 1:size(xk,1)
        w = ws(jx,:); ykj = yk(jx,:); xkj = xk(jx,:);
        % h1 = @(x) lamdbak*sum(abs(w)) + prod(lambdak*sign(w),x - w) - M*norm(x-w,'fro')^2/2 + norm(x-ykj,'fro')^2/(2*h) - (n-alpha)*delta/2;  %(n-alpha)*delpha = 0
        mu = (1/h-M); %variance of h1
        % rejection sampling
        Xnew = (ykj/h - M*w - lambdak*sign(w))/mu + randn(1,d)/sqrt(mu); %samples drawed from h1
        U = rand;
        rej = 0;
        rej_pro = exp(-fg_den(Xnew))*pf(Xnew,xkj)/(exp(-fg_den(xkj))*pf(xkj,Xnew));
        while U > rej_pro 
            Xnew = (ykj/h - M*w - lambdak*sign(w))/mu + randn(1,d)/sqrt(mu); %samples drawed from h1
            U = rand;
            rej = rej + 1;
            rej_pro = exp(-fg_den(Xnew))*pf(Xnew,xkj)/(exp(-fg_den(xkj))*pf(xkj,Xnew));
            if rej > 10
                break;
            end
        end
        output(jx,:) = Xnew;
    end
end  
