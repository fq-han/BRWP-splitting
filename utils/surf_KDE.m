function z = surf_KDE(xk,bandwidth,bdL,bdR)
    xx = linspace(bdL,bdR,200);
    [x1,x2] = meshgrid(xx,xx);
    nx = size(xk,1);
    z = 0;
    for jx = 1:nx
        z = z + exp(-((xk(jx,1)-x1(:)).^2+(xk(jx,2)-x2(:)).^2)./(2*bandwidth));
    end
    z = z./nx;
    surf(x1,x2,reshape(z,size(x1)),'EdgeColor', 'none', 'FaceAlpha', 1, 'FaceColor','interp'); view(45,20); 
end