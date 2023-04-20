%% visualization of leverage score distributions
red = [0.8500, 0.3250, 0.0980];
blue = [0 0.4470 0.7410];
gray = .5*[1 1 1];
% gaussian
falpha = 1;
lim = 5
step = .1
[Xsamp,Ysamp] = meshgrid(-lim:step:lim);
Z = exp(-(Xsamp.^2 + Ysamp.^2));

xvals = -lim:step:lim;
yvals = -lim:step:lim;
deg = 8;
H = zeros(length(xvals),8);
for i=1:deg
    H(:,i) = hermite(i-1,xvals);
end
plot(xvals, H(:,7))

N = length(xvals)*length(yvals);
H2d = zeros(N,deg*(deg-1)/2);
c = 1;
for i = 1:deg
    for j = 1:(deg-i+1)
        H2d(:,c) = reshape(H(:,i)*H(:,j)',N,1);
        c = c+1;
    end
end
sG = reshape(Z,length(xvals)*length(yvals),1);
U = orth(sqrt(sG).*H2d);
levs = sum(U.^2,2);
levsSquare = reshape(levs,length(xvals),length(yvals));
% imagesc(levsSquare);
% colorbar

figure();
Zs = (1/step^2)*Z/sum(sum(Z));
sPlot = surf(Xsamp,Ysamp,Zs,'FaceAlpha',falpha,'FaceColor',blue)
ylim([-lim,lim])
xlim([-lim,lim])
zlim([0,max(max(Zs))*1])
exportgraphics(gca,'gauss_dist.png','Resolution',600) 


figure();
levsSquareS = (1/step^2)*levsSquare/sum(sum(levsSquare));
sPlot = surf(Xsamp,Ysamp,levsSquareS,'FaceAlpha',falpha,'FaceColor',blue)
ylim([-lim,lim])
xlim([-lim,lim])
zlim([0,max(max(Zs))*.33])
exportgraphics(gca,'gauss_levs.png','Resolution',600) 

% uniform
lim = 2
step = .05
[Xsamp,Ysamp] = meshgrid(-lim:step:lim);
Z = (max(abs(Xsamp),abs(Ysamp)) <= 1);
xvals = -lim:step:lim;
yvals = -lim:step:lim;

deg = 8;
T = zeros(length(xvals),8);
T(:,1) = abs(xvals) <= 1;
T(:,2) = T(:,1).*xvals';
for i = 3:deg
    T(:,i) = 2*T(:,2).*T(:,i-1) - T(:,i-2);
end

N = length(xvals)*length(yvals);
T2d = zeros(N,deg*(deg-1)/2);
c = 1;
for i = 1:deg
    for j = 1:(deg-i+1)
        T2d(:,c) = reshape(T(:,i)*T(:,j)',N,1);
        c = c+1;
    end
end
U = orth(T2d);
levs = sum(U.^2,2);
levsSquare = reshape(levs,length(xvals),length(yvals));
% imagesc(levsSquare);
% colorbar

figure();
levsSquareS = (1/step^2)*levsSquare/sum(sum(levsSquare));
sPlot = surf(Xsamp,Ysamp,levsSquareS,'FaceAlpha',falpha,'FaceColor',red)
ylim([-lim,lim])
xlim([-lim,lim])
zlim([0,max(max(levsSquareS))*1])
exportgraphics(gca,'uniform_levs.png','Resolution',600) 

figure();
Zs = (1/step^2)*Z/sum(sum(Z));
sPlot = surf(Xsamp,Ysamp,Zs,'FaceAlpha',falpha,'FaceColor',red)
ylim([-lim,lim])
xlim([-lim,lim])
zlim([0,max(max(levsSquareS))*.33])
exportgraphics(gca,'uniform_dist.png','Resolution',600) 

% cauchy
lim = 20
step = .1
[Xsamp,Ysamp] = meshgrid(-lim:step:lim);
% Z = exp(-sqrt((Xsamp).^2 + abs(Ysamp).^2));
Z = (1./(1 + Ysamp.^2))*(1./(1 + Xsamp.^2));
xvals = -lim:step:lim;
yvals = -lim:step:lim;

deg = 8;
H = zeros(length(xvals),8);
for i=1:deg
    H(:,i) = hermite(i-1,xvals);
end
plot(xvals, H(:,7))

N = length(xvals)*length(yvals);
H2d = zeros(N,deg*(deg-1)/2);
c = 1;
for i = 1:deg
    for j = 1:(deg-i+1)
        H2d(:,c) = reshape(H(:,i)*H(:,j)',N,1);
        c = c+1;
    end
end
sC = reshape(Z,length(xvals)*length(yvals),1);
U = orth(sqrt(sC).*H2d);
levs = sum(U.^2,2);
levsSquare = reshape(levs,length(xvals),length(yvals));


plotlim = 6;
Z = Z.*(max(abs(Xsamp),abs(Xsamp)) < plotlim);
figure();
Zs = (1/step^2)*Z/sum(sum(Z));
sPlot = surf(Xsamp,Ysamp,Zs,'FaceAlpha',falpha,'FaceColor',gray)
ylim([-plotlim,plotlim])
xlim([-plotlim,plotlim])
zlim([0,max(max(Zs))*1])
exportgraphics(gca,'cauchy_dist.png','Resolution',600) 

figure();
levsSquare = levsSquare.*(max(abs(Xsamp),abs(Xsamp)) < plotlim);
levsSquareS = (1/step^2)*levsSquare/sum(sum(levsSquare));
sPlot = surf(Xsamp,Ysamp,levsSquareS,'FaceAlpha',falpha,'FaceColor',gray)
ylim([-plotlim,plotlim])
xlim([-plotlim,plotlim])
zlim([0,max(max(Zs))*.33])
exportgraphics(gca,'cauchy_levs.png','Resolution',600) 

