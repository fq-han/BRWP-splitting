function [g,dg,xex,prhos1,prhos2] = Mixture_example(example_idx,d,lambdak)
    dg = cell(1,d); xex = zeros(1,d);   prhos1 = 0; prhos2 = 0;
    switch example_idx     
        case 2
            %% distribution is Mixed Gaussian
            var_gau = 4; sd_gau = sqrt(var_gau); 
            a1 = zeros(1,d); a2= a1; a3 = a2; a4 = a3;
            S1 = eye(d)./(2*var_gau); S2 = eye(d)./(2*var_gau); S3 = eye(d)./(2*var_gau); S4 = eye(d)./(2*var_gau);
            a1(1:2) = [4,-2]; a2(1:2) = [-2,-4];  a3(1:2) = [-4,2];  a4(1:2) = [2,-4]; 
            as = [a1;a2;a3;a4].*2; 
            % as(:,3:floor(d/4)) = randn(4,floor(d/4)-2);
            cs = [1,1,1,1]; cs = cs./sum(cs); 
            g = @(xk)-log(cs(1).*exp(-sum((xk-as(1,:))*S1.*(xk-as(1,:)),2))+cs(2).*exp(-sum((xk-as(2,:))*S2.*(xk-as(2,:)),2))+cs(3).*exp(-sum((xk-as(3,:))*S3.*(xk-as(3,:)),2))+cs(4).*exp(-sum((xk-as(4,:))*S4.*(xk-as(4,:)),2)));
            env = @(xk)cs(1).*exp(-sum((xk-as(1,:))*S1.*(xk-as(1,:)),2))+cs(2).*exp(-sum((xk-as(2,:))*S2.*(xk-as(2,:)),2))...
                +cs(3).*exp(-sum((xk-as(3,:))*S3.*(xk-as(3,:)),2))+cs(4).*exp(-sum((xk-as(4,:))*S4.*(xk-as(4,:)),2));
            for jd = 1:d
                dg{jd} = @(xk)2.*(cs(1).*(xk-as(1,:))*S1(:,jd).*exp(-sum((xk-as(1,:))*S1.*(xk-as(1,:)),2))+cs(2).*(xk-as(2,:))*S2(:,jd).*exp(-sum((xk-as(2,:))*S2.*(xk-as(2,:)),2))...
                      +cs(3).*(xk-as(3,:))*S3(:,jd).*exp(-sum((xk-as(3,:))*S3.*(xk-as(3,:)),2))+cs(4).*(xk-as(4,:))*S4(:,jd).*exp(-sum((xk-as(4,:))*S4.*(xk-as(4,:)),2)))./env(xk);
            end
            xx = -20:0.01:20; 
            px = ones(4,d);
            for jn = 1:4
                for jd = 2:d
                    s1 = -(as(jn,jd)-lambdak*var_gau)/(sqrt(2*var_gau)); 
                    s2 = -(as(jn,jd)+lambdak*var_gau)/(sqrt(2*var_gau));
                    px(jn,jd) = exp(-(as(jn,jd)^2-(as(jn,jd)-lambdak*sd_gau)^2)/(2*var_gau))*gauss_int(s1,inf) + exp(-(as(jn,jd)^2-(as(jn,jd)+lambdak*sd_gau)^2)/(2*var_gau))*gauss_int(-inf,s2);
                 end
            end
            px = prod(px,2); prhos = 0;
            for jn = 1:4
                prhos = prhos + px(jn).*cs(jn).*exp(-(xx-as(jn,1)).^2./(2*var_gau)-lambdak*abs(xx));
            end
            prhos1 = prhos./sum(prhos(:).*0.01);
            
            px = ones(4,d-1);
            for jn = 1:4
                % for jd = [1,3:d]
                for jd = 1:d-1 
                s1 = -(as(jn,jd)-lambdak*var_gau)/(sqrt(2*var_gau)); 
                s2 = -(as(jn,jd)+lambdak*var_gau)/(sqrt(2*var_gau));
                px(jn,jd) = exp(-(as(jn,jd)^2-(as(jn,jd)-lambdak*sd_gau)^2)/(2*var_gau))*gauss_int(s1,inf) + exp(-(as(jn,jd)^2-(as(jn,jd)+lambdak*sd_gau)^2)/(2*var_gau))*gauss_int(-inf,s2);
                end
            end
            px = prod(px,2); prhos = 0;
            for jn = 1:4
                prhos = prhos + px(jn).*cs(jn).*exp(-(xx-as(jn,d)).^2./(2*var_gau)-lambdak*abs(xx));
            end
            prhos2 = prhos./sum(prhos(:).*0.01);
        case 21
             %% distribution is Mixed Gaussian for general log(d) inclusions
            var_gau = 4; sd_gau = sqrt(var_gau); nmodel =8;
            as = zeros(nmodel,d); S = eye(d)./(2*var_gau);
            rng(10);
            as(1:nmodel,1:nmodel) = (rand(nmodel)-0.5)*10; %this is the center location
            cs = ones(nmodel,1)/nmodel; %this is the coefficients

            g = @(xk)-log(gaux(xk,as,cs,S,nmodel));
            env = @(xk)gaux(xk,as,cs,S,nmodel);
 
           for jd = 1:d
                dg{jd} = @(xk)2.*(dgaux(xk,as,cs,S,S(:,jd),nmodel))./env(xk);
           end

            %% computing marginal distirbution
            hxx = 0.01; xx = -20:hxx:20; 
            px = ones(nmodel,d-1);
            for jn = 1:nmodel
                for jd = 2:d
                s1 = -(as(jn,jd)-lambdak*var_gau)/(sqrt(2*var_gau)); 
                s2 = -(as(jn,jd)+lambdak*var_gau)/(sqrt(2*var_gau));
                px(jn,jd) = exp(-(as(jn,jd)^2-(as(jn,jd)-lambdak*sd_gau)^2)/(2*var_gau))*gauss_int(s1,inf) + exp(-(as(jn,jd)^2-(as(jn,jd)+lambdak*sd_gau)^2)/(2*var_gau))*gauss_int(-inf,s2);
                end
            end
            px = prod(px,2); prhos = 0;
            for jn = 1:nmodel
                prhos = prhos + px(jn).*cs(jn).*exp(-(xx-as(jn,1)).^2./(2*var_gau)-lambdak*abs(xx));
            end
            prhos1 = prhos./sum(prhos(:).*hxx);
            
            px = ones(nmodel,d-1);
            for jn = 1:nmodel
                % for jd = [1,3:d]
                for jd = 1:d-1 
                s1 = -(as(jn,jd)-lambdak*var_gau)/(sqrt(2*var_gau)); 
                s2 = -(as(jn,jd)+lambdak*var_gau)/(sqrt(2*var_gau));
                px(jn,jd) = exp(-(as(jn,jd)^2-(as(jn,jd)-lambdak*sd_gau)^2)/(2*var_gau))*gauss_int(s1,inf) + exp(-(as(jn,jd)^2-(as(jn,jd)+lambdak*sd_gau)^2)/(2*var_gau))*gauss_int(-inf,s2);
                end
            end
            px = prod(px,2); prhos = 0;
            for jn = 1:nmodel
                prhos = prhos + px(jn).*cs(jn).*exp(-(xx-as(jn,d)).^2./(2*var_gau)-lambdak*abs(xx));
            end
            prhos2 = prhos./sum(prhos(:).*0.01);
    end
end

function dgall = vec_fun_dg(xk,dg,d)
    dgall = zeros(size(xk));
    for jd = 1:d
        tmp = dg{jd}(xk);
        dgall(:,jd) = tmp(1:size(xk,1));
    end
end

function val = gauss_int(a,b)
    maxab = max([abs(a);abs(b)],[],1); minab = min([abs(a);abs(b)],[],1);
    val = erf(maxab) - sign(a).*sign(b).*erf(minab);
    val = val.*sqrt(pi)/2;
end

function expdgval = dgaux(xk,as,cs,S,Sjd,nmodel)
    expdgval = 0; 
    for jmodel = 1:nmodel
        expdgval = expdgval + cs(jmodel).*(xk-as(jmodel,:))*Sjd.*exp(-sum((xk-as(jmodel,:))*S.*(xk-as(jmodel,:)),2));
    end
end

function exgval = gaux(xk,as,cs,S,nmodel)
    exgval = 0; 
    for jmodel = 1:nmodel
        exgval = exgval + (cs(jmodel).*exp(-sum((xk-as(jmodel,:))*S.*(xk-as(jmodel,:)),2)));
    end
end