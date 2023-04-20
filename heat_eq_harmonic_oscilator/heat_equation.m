% heat equation
% example taken from http://courses.washington.edu/amath581/PDEtool.pdf

%% varying time and frequency of starting condition
freq = 13;
u0 = @(x) sin(freq*pi*x) + 1;

m = 0;
x = linspace(0,1,100);
tvals = 800;
t = linspace(0,3,tvals);
u = pdepe(m,'pdex1pde',u0,'pdex1bc',x,t);
surf(x,t,u)

fvals = 0:.05:5;
qoifull = zeros(length(fvals),tvals);
for i = 1:length(fvals)
    freq = fvals(i);
    u0 = @(x) sin(freq*pi*x) + 1;
    u = pdepe(m,'pdex1pde',u0,'pdex1bc',x,t);
    qoifull(i,:) = max(u')';
end
figure();
% qoifull = qoifull - min(min(qoifull));
imagesc(qoifull)
colorbar

%% constructing regression matrix
% our acutally degree is deg - 1
deg = 12;
Pw = zeros(length(fvals),deg);
Pk = zeros(length(t),deg);
Pw(:,1) = ones(length(fvals),1);
Pk(:,1) = ones(length(t),1);
Pw(:,2) = linspace(-1,1,length(fvals));
Pk(:,2) = linspace(-1,1,length(t));
for i = 3:deg
    Pw(:,i) = 2*Pw(:,2).*Pw(:,i-1) - Pw(:,i-2);
    Pk(:,i) = 2*Pk(:,2).*Pk(:,i-1) - Pk(:,i-2);
end

N = length(fvals)*length(t);
P2d = zeros(N,deg*(deg-1)/2);
c = 1;
for i = 1:deg
    for j = 1:(deg-i+1)
        P2d(:,c) = reshape(Pw(:,i)*Pk(:,j)',N,1);
        c = c+1;
    end
end

U = orth(P2d);
levs = sum(U.^2,2);
levsSquare = reshape(levs,length(fvals),length(t));

%% generate final plots: generates image 2b and 2c
figure();
colormap(turbo);
imagesc(flipud(qoifull));
colorbar
% caxis([0 .5]);
set(gca,'fontsize',14);
set(gca,'TickLabelInterpreter','latex');
xlim([0,length(t)])
ylim([0,length(fvals)])
set(gca,'fontsize',14);
set(gca,'TickLabelInterpreter','latex');
xind = 1:floor((length(t)/10)):length(t);
xind = [xind,length(t)];
yind = 1:floor((length(fvals)/10)):length(fvals);
% yind = [yind,length(fvals)]
set(gca,'Xtick',xind,'XTickLabel',round(t(xind),2));
% xtickformat('$%,.3f')
set(gca,'Ytick',yind,'YTickLabel',flip(fvals(yind)));
xlabel('time , $t$','FontSize',30,'interpreter','latex');
ylabel('starting frequency, $f$','FontSize',30,'interpreter','latex');
exportgraphics(gca,'heat_eq_true.png','Resolution',600) 

dotsize = 150;

%% nonlinearity
f = @(x) exp(x);
fprime = @(x) exp(x);

nsamps = 120;
qoivec = reshape(qoifull,N,1);
% uniform random samples
indunif = randi(N,nsamps,1);
[urows,ucols] = ind2sub([length(fvals) length(t)],indunif);
A = P2d(indunif,:);
b = qoivec(indunif);
x = grad_descent(A,b,ones(size(b)),f,fprime,1000,.1);
qoi_fit_uniform = arrayfun(f,P2d*x);
qoi_fit_uniform = reshape(qoi_fit_uniform,length(fvals),length(t));
figure();
colormap(turbo);
imagesc(flipud(qoi_fit_uniform));
colorbar
hold();
mean(mean((qoi_fit_uniform - qoifull).^2))
xlim([0,length(t)])
ylim([0,length(fvals)])
set(gca,'fontsize',14);
set(gca,'TickLabelInterpreter','latex');
xind = 1:floor((length(t)/10)):length(t);
xind = [xind,length(t)];
yind = 1:floor((length(fvals)/10)):length(fvals);
% yind = [yind,length(fvals)]
set(gca,'Xtick',xind,'XTickLabel',round(t(xind),2));
% xtickformat('$%,.3f')
set(gca,'Ytick',yind,'YTickLabel',flip(fvals(yind)));
xlabel('time , $t$','FontSize',30,'interpreter','latex');
ylabel('starting frequency, $f$','FontSize',30,'interpreter','latex');
% title('Uniform Sampling Approximation','FontSize',30,'interpreter','latex');
exportgraphics(gca,'heat_eq_uniform_exp2.png','Resolution',600) 


% leverage score samples
indlev = randsample(N,nsamps,true,levs);
[lrows,lcols] = ind2sub([length(fvals) length(t)],indlev);
weight_vec = 1./levs;
sP2d = P2d;
sqoivec = qoivec;
A = sP2d(indlev,:);
b = sqoivec(indlev);
x = grad_descent(A,b,weight_vec(indlev),f,fprime,1000,.1);
qoi_fit_lev = arrayfun(f,P2d*x);
qoi_fit_lev = reshape(qoi_fit_lev,length(fvals),length(t));
figure(); 
colormap(turbo);
imagesc(real(flipud(qoi_fit_lev)));
colorbar
hold();

mean(mean((qoi_fit_lev - qoifull).^2))
xlim([0,length(t)])
ylim([0,length(fvals)])
set(gca,'fontsize',14);
set(gca,'TickLabelInterpreter','latex');
xind = 1:floor((length(t)/10)):length(t);
xind = [xind,length(t)];
yind = 1:floor((length(fvals)/10)):length(fvals);
set(gca,'Xtick',xind,'XTickLabel',round(t(xind),2));
set(gca,'Ytick',yind,'YTickLabel',flip(fvals(yind)));
xlabel('time , $t$','FontSize',30,'interpreter','latex');
ylabel('starting frequency, $f$','FontSize',30,'interpreter','latex');
exportgraphics(gca,'heat_eq_lev_exp2.png','Resolution',600) 




%% computing l2 errors
trials = 100;
nsampvals = 30:20:1000;
errsunif = zeros(trials,length(nsampvals));
errslev = zeros(trials,length(nsampvals));

for i = 1:length(nsampvals)
    nsamps = nsampvals(i)
    for j = 1:trials
        qoivec = reshape(qoifull,N,1);
        % uniform random samples
        indunif = randi(N,nsamps,1);
        [urows,ucols] = ind2sub([length(fvals) length(t)],indunif);
        A = P2d(indunif,:);
        b = qoivec(indunif);
        x = grad_descent(A,b,ones(size(b)),f,fprime,1000,.1);
        qoi_fit_uniform = arrayfun(f,P2d*x);
        qoi_fit_uniform = reshape(qoi_fit_uniform,length(fvals),length(t));
        errsunif(j,i) = mean(mean((qoi_fit_uniform - qoifull).^2));
        
        % leverage samples
        indlev = randsample(N,nsamps,true,levs);
        [lrows,lcols] = ind2sub([length(fvals) length(t)],indlev);
        weight_vec = (1./(levs));
        sP2d = P2d;
        sqoivec = qoivec;
        A = sP2d(indlev,:);
        b = sqoivec(indlev);
        x = grad_descent(A,b,weight_vec(indlev),f,fprime,1000,.001);
        qoi_fit_lev = arrayfun(f,P2d*x);
        qoi_fit_lev = reshape(qoi_fit_lev,length(fvals),length(t));
        errslev(j,i) = mean(mean((qoi_fit_lev - qoifull).^2)); 
    end
end

mqoi = mean(mean(qoifull.^2));

unifsort = sort(errsunif)/mqoi;
unifmedian = unifsort(50,:);
uniflower = unifsort(25,:);
unifupper = unifsort(75,:);

levsort = sort(errslev)/mqoi;
levmedian = levsort(50,:);
levlower = levsort(25,:);
levupper = levsort(75,:);

red = [0.8500, 0.3250, 0.0980];
blue = [0 0.4470 0.7410];

% quantitative plot
c = 26;
figure(); hold();
plot(nsampvals(1:c), unifmedian(1:c), 'Color', red, 'LineWidth', 2)
plot(nsampvals(1:c), levmedian(1:c),'Color', blue, 'LineWidth', 2)

x2 = [nsampvals(1:c), fliplr(nsampvals(1:c))];
inBetween = [uniflower(1:c), fliplr(unifupper(1:c))];
h = fill(x2,inBetween,red);
set(h,'facealpha',.1)
set(h,'edgealpha',.1)

x2 = [nsampvals(1:c), fliplr(nsampvals(1:c))];
inBetween = [levlower(1:c), fliplr(levupper(1:c))];
h = fill(x2,inBetween,blue);
set(h,'facealpha',.1)
set(h,'edgealpha',.1)

set(gca,'fontsize',18);
set(gca,'TickLabelInterpreter','latex');
set(gca, 'YScale', 'log');
% set(gca, 'XScale', 'log');
xlim([nsampvals(1),nsampvals(c)])
ylim([10^-4,10^0])

legend({'Uniform Sampling', 'Leverage Sampling'}, 'FontSize',20,'interpreter','latex','Location', 'northeast');
xlabel('number of samples','FontSize',24,'interpreter','latex');
ylabel('relative mean squared error','FontSize',24,'interpreter','latex');
exportgraphics(gca,'pde_unif_lev_compare_all_exp.png','Resolution',600) 



        
