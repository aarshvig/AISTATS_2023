% damped harmonic oscillator with periodic driving force
% example taken from https://www5.in.tum.de/lehre/vorlesungen/algo_uq/ss18/06_polynomial_chaos.pdf

%% changing single variabe: the forcing frequency w. Just a quick test here
% parameter setup
c = 0.5;
k = 2.0;
f = 0.5;
w = .8;
p = [c,k,f,w];
% initial condition 
yinit = [.5 0];

% time domain setup
tmax = 20;
dt = 0.01;

[t,y] = ode45(@(t,y) odemodel(t,y,p),[0 tmax],yinit);
figure();
plot(t,y(:,1),t,y(:,2))

wvals = 0:.001:2;
qoifull = zeros(size(wvals));
for i = 1:length(wvals)
    p(4) = wvals(i);
    [t,y] = ode45(@(t,y) odemodel(t,y,p),[0 tmax],yinit);
    qoi(i) = max(y(:,1));
end
plot(wvals,qoi)

deg = 10;
P = zeros(length(wvals),deg);
P(:,1) = ones(length(wvals),1);
P(:,2) = wvals-1;
for i = 3:deg
    P(:,i) = 2*P(:,2).*P(:,i-1) - P(:,i-2);
end

nsamps = 20;
% uniform random samples
indunif = randi(length(wvals),nsamps,1);
qoi_fit_uniform = P*(pinv(P(indunif,:))*qoi(indunif)');
figure();
plot(wvals,qoi,wvals,qoi_fit_uniform);


% leverage score samples
U = orth(P);
levs = sum(U.^2,2);
indlev = randsample(length(wvals),nsamps,true,levs);
sP = (1./sqrt(levs)).*P;
sqoi = (1./sqrt(levs))'.*qoi;
qoi_fit_lev = P*(pinv(sP(indlev,:))*sqoi(indlev)');
figure();
plot(wvals,qoi,wvals,qoi_fit_lev);


%% changing multiple variable: the forcing frequency w and spring constant k
% parameter setup
c = 0.5;
k = 2.0;
f = 0.5;
w = .8;
p = [c,k,f,w];
% initial condition 
yinit = [.5 0];

% time domain setup
tmax = 20;
dt = 0.01;


kvals = 1:.01:3;
wvals = 0:.01:2;
qoifull = zeros(length(wvals),length(kvals));
for i = 1:length(wvals)
    for j = 1:length(kvals)
        p(4) = wvals(i);
        p(2) = kvals(j);
        [t,y] = ode45(@(t,y) odemodel(t,y,p),[0 tmax],yinit);
        qoifull(i,j) = max(y(:,1));
    end
end
imagesc(qoifull)
colorbar

% our acutally degree is deg - 1
deg = 13;
Pw = zeros(length(wvals),deg);
Pk = zeros(length(kvals),deg);
Pw(:,1) = ones(length(wvals),1);
Pk(:,1) = ones(length(kvals),1);
Pw(:,2) = wvals-1;
Pk(:,2) = kvals-2;
for i = 3:deg
    Pw(:,i) = 2*Pw(:,2).*Pw(:,i-1) - Pw(:,i-2);
    Pk(:,i) = 2*Pk(:,2).*Pk(:,i-1) - Pk(:,i-2);
end

N = length(wvals)*length(kvals);
P2d = zeros(N,deg*(deg-1)/2);
c = 1;
for i = 1:deg
    for j = 1:(deg-i+1)
        P2d(:,c) = reshape(Pw(:,i)*Pw(:,j)',N,1);
        c = c+1;
    end
end

U = orth(P2d);
levs = sum(U.^2,2);
levsSquare = reshape(levs,length(wvals),length(kvals));
imagesc(levsSquare);
colorbar

% what does sampling by leverage scores look like?
msamp = .05*rand(length(wvals),length(kvals)) < levsSquare;
spy(msamp)

qoifull = qoifull-.5;

%% generate final plots: generates image 2b and 2c
figure();
imagesc(flipud(qoifull));
colorbar
caxis([0 .5]);
set(gca,'fontsize',14);
set(gca,'TickLabelInterpreter','latex');
xind = 1:floor((length(kvals)/10)):length(kvals);
yind = 1:floor((length(wvals)/10)):length(wvals);
set(gca,'Xtick',xind,'XTickLabel',kvals(xind));
set(gca,'Ytick',yind,'YTickLabel',flip(wvals(yind)));
xlabel('spring constant, $k$','FontSize',30,'interpreter','latex');
ylabel('driving frequency, $\omega$','FontSize',30,'interpreter','latex');
% title('True Max Displacement','FontSize',30,'interpreter','latex');
exportgraphics(gca,'spring_displace_true.png','Resolution',600) 

dotsize = 150;

nsamps = 200;
qoivec = reshape(qoifull,N,1);
% uniform random samples
indunif = randi(N,nsamps,1);
[urows,ucols] = ind2sub([length(wvals) length(kvals)],indunif);
A = P2d(indunif,:);
b = qoivec(indunif);
f = @(x) max(x,0);
fprime = @(x) (x>=0);
x = grad_descent(A,b,f,fprime,10000,.1);
qoi_fit_uniform = f(P2d*x);
qoi_fit_uniform = reshape(qoi_fit_uniform,length(wvals),length(kvals));
figure();
imagesc(flipud(qoi_fit_uniform));
colorbar
caxis([0 .5]);
hold();
mean(mean((qoi_fit_uniform - qoifull).^2))
xlim([0,length(wvals)])
ylim([0,length(kvals)])
set(gca,'fontsize',14);
set(gca,'TickLabelInterpreter','latex');
xind = 1:floor((length(kvals)/10)):length(kvals);
yind = 1:floor((length(wvals)/10)):length(wvals);
set(gca,'Xtick',xind,'XTickLabel',kvals(xind));
set(gca,'Ytick',yind,'YTickLabel',flip(wvals(yind)));
xlabel('spring constant, $k$','FontSize',30,'interpreter','latex');
ylabel('driving frequency, $\omega$','FontSize',30,'interpreter','latex');
% title('Uniform Sampling Approximation','FontSize',30,'interpreter','latex');
exportgraphics(gca,'spring_displace_unif.png','Resolution',600) 


% leverage score samples
indlev = randsample(N,nsamps,true,levs);
[lrows,lcols] = ind2sub([length(wvals) length(kvals)],indlev);
sP2d = (1./sqrt(levs)).*P2d;
sqoivec = (1./sqrt(levs)).*qoivec;
A = sP2d(indlev,:);
b = sqoivec(indlev);
f = @(x) max(x,0);
fprime = @(x) (x>=0);
x = grad_descent(A,b,f,fprime,10000,.1);
qoi_fit_lev = f(P2d*x);
qoi_fit_lev = reshape(qoi_fit_lev,length(wvals),length(kvals));
figure(); 
imagesc(flipud(qoi_fit_lev));
colorbar
caxis([0 .5]);
hold();

% scatter(lrows,lcols,dotsize,'.k');
mean(mean((qoi_fit_lev - qoifull).^2))
xlim([0,length(wvals)])
ylim([0,length(kvals)])
set(gca,'fontsize',14);
set(gca,'TickLabelInterpreter','latex');
xind = 1:floor((length(kvals)/10)):length(kvals);
yind = 1:floor((length(wvals)/10)):length(wvals);
set(gca,'Xtick',xind,'XTickLabel',kvals(xind));
set(gca,'Ytick',yind,'YTickLabel',flip(wvals(yind)));
xlabel('spring constant, $k$','FontSize',30,'interpreter','latex');
ylabel('driving frequency, $\omega$','FontSize',30,'interpreter','latex');
% title('Leverage Sampling Approximation','FontSize',30,'interpreter','latex');
exportgraphics(gca,'spring_displace_lev.png','Resolution',600) 

%% computing l2 errors
trials = 100;
nsampvals = 130:20:1000
errsunif = zeros(trials,length(nsampvals));
errslev = zeros(trials,length(nsampvals));
% errsquasi = zeros(trials,length(nsampvals));
% errsquasiunif = zeros(trials,length(nsampvals));

%quasi random sequence
% p = haltonset(2,'Skip',1e3,'Leap',1e2);
% p = scramble(p,'RR2');

f = @(x) max(x,0);
fprime = @(x) (x>=0);
for i = 1:length(nsampvals)
% for i = 1:length(nsampvals)
    nsamps = nsampvals(i)
    for j = 1:trials
        qoivec = reshape(qoifull,N,1);
        % uniform random samples
        indunif = randi(N,nsamps,1);
        [urows,ucols] = ind2sub([length(wvals) length(kvals)],indunif);
        A = P2d(indunif,:);
        b = qoivec(indunif);
        f = @(x) max(x,0);
        fprime = @(x) (x>=0);
        x = grad_descent(A,b,f,fprime,10000,.1);
        qoi_fit_uniform = f(P2d*x);
        qoi_fit_uniform = reshape(qoi_fit_uniform,length(wvals),length(kvals));
        errsunif(j,i) = mean(mean((qoi_fit_uniform - qoifull).^2));
        
        indlev = randsample(N,nsamps,true,levs);
        [lrows,lcols] = ind2sub([length(wvals) length(kvals)],indlev);
        sP2d = (1./sqrt(levs)).*P2d;
        sqoivec = (1./sqrt(levs)).*qoivec;
        A = sP2d(indlev,:);
        b = sqoivec(indlev);
        x = grad_descent(A,b,f,fprime,10000,.1);
        qoi_fit_lev = f(P2d*x);
        qoi_fit_lev = reshape(qoi_fit_lev,length(wvals),length(kvals));
        errslev(j,i) = mean(mean((qoi_fit_lev - qoifull).^2)); 
%         xyval = round(2*p(c:nsamps+c,:)-1,2);
%         indquasi = int64((xyval+1)*100) + 1;
%         qrows = indquasi(:,1); qcols = indquasi(:,2);
%         indquasi = sub2ind([length(wvals) length(kvals)],indquasi(:,1),indquasi(:,2));
%         qoi_fit_quasi = P2d*(pinv(P2d(indquasi,:))*qoivec(indquasi));
%         qoi_fit_quasi = reshape(qoi_fit_quasi,length(wvals),length(kvals));
%         errsquasiunif(j,i) = mean(mean((qoi_fit_quasi - qoifull).^2));
%         
%         xyval = round(cos(p(c:nsamps+c,:)*pi),2);
%         c = c + nsamps;
%         indquasi = int64((xyval+1)*100) + 1;
%         qrows = indquasi(:,1); qcols = indquasi(:,2);
%         indquasi = sub2ind([length(wvals) length(kvals)],indquasi(:,1),indquasi(:,2));
%         qoi_fit_quasi = P2d*(pinv(sP2d(indquasi,:))*sqoivec(indquasi));
%         qoi_fit_quasi = reshape(qoi_fit_quasi,length(wvals),length(kvals));
%         errsquasi(j,i) = mean(mean((qoi_fit_quasi - qoifull).^2));   
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

% quasisort = sort(errsquasi)/mqoi;
% quasimedian = quasisort(50,:)/mqoi;
% quasilower = quasisort(25,:)/mqoi;
% quasiupper = quasisort(75,:)/mqoi;
% 
% quasiunifsort = sort(errsquasiunif)/mqoi;
% quasiunifmedian = quasiunifsort(50,:)/mqoi;
% quasiuniflower = quasiunifsort(25,:)/mqoi;
% quasiunifupper = quasiunifsort(75,:)/mqoi;


% compute best possible error
% qoi_fit_best = P2d*(pinv(P2d)*qoivec);
% qoi_fit_best = reshape(qoi_fit_best,length(wvals),length(kvals));
% best_error = mean(mean((qoi_fit_best - qoifull).^2))/mqoi;

red = [0.8500, 0.3250, 0.0980];
blue = [0 0.4470 0.7410];

mqoi = mean(mean(qoifull.^2));

% quantitative plot
c = 21;
figure(); hold();
plot(nsampvals(1:c), unifmedian(1:c), 'Color', red, 'LineWidth', 2)
% plot(nsampvals(1:c), quasiunifmedian(1:c),'-.', 'Color', red,'LineWidth', 2)
plot(nsampvals(1:c), levmedian(1:c),'Color', blue, 'LineWidth', 2)
% plot(nsampvals(1:c), quasimedian(1:c),'-.', 'Color', blue,'LineWidth', 2)
% plot(nsampvals(1:c), best_error*ones(1,c),'--k', 'LineWidth', 2)

x2 = [nsampvals(1:c), fliplr(nsampvals(1:c))];
inBetween = [uniflower(1:c), fliplr(unifupper(1:c))];
h = fill(x2,inBetween,red);
set(h,'facealpha',.1)
set(h,'edgealpha',.1)


% x2 = [nsampvals(1:c), fliplr(nsampvals(1:c))];
% inBetween = [quasiuniflower(1:c), fliplr(quasiunifupper(1:c))];
% h = fill(x2,inBetween,red);
% set(h,'facealpha',.1)
% set(h,'edgealpha',.1)

x2 = [nsampvals(1:c), fliplr(nsampvals(1:c))];
inBetween = [levlower(1:c), fliplr(levupper(1:c))];
h = fill(x2,inBetween,blue);
set(h,'facealpha',.1)
set(h,'edgealpha',.1)

% x2 = [nsampvals(1:c), fliplr(nsampvals(1:c))];
% inBetween = [quasilower(1:c), fliplr(quasiupper(1:c))];
% h = fill(x2,inBetween,'b');
% set(h,'facealpha',.1)
% set(h,'edgealpha',.1)


set(gca,'fontsize',18);
set(gca,'TickLabelInterpreter','latex');
set(gca, 'YScale', 'log');
% set(gca, 'XScale', 'log');
xlim([nsampvals(1),nsampvals(c)])
ylim([10^-3,10^1.5])

legend({'Uniform Sampling', 'Leverage Sampling'}, 'FontSize',20,'interpreter','latex','Location', 'northeast');
xlabel('number of samples','FontSize',24,'interpreter','latex');
ylabel('relative mean squared error','FontSize',24,'interpreter','latex');
exportgraphics(gca,'unif_lev_compare_all.png','Resolution',600) 


% visualizing sampling patterns
dotsize = 200

% quasi-random
p = haltonset(2,'Skip',1e3,'Leap',1e2)
p = scramble(p,'RR2')
X0 = p(1:500,:);
scatter(cos(X0(:,1)*pi),cos(X0(:,2)*pi),dotsize,'r.')
exportgraphics(gca,'quasi_samps.png','Resolution',600) 

% quasi-random unfiform
p = haltonset(2,'Skip',1e3,'Leap',1e2)
p = scramble(p,'RR2')
X0 = p(1:500,:);
scatter(X0(:,1)*2-1,X0(:,2)*2-1,dotsize,'.','MarkerEdgeColor',[0 .5 0],'MarkerFaceColor',[0 .5 0])
exportgraphics(gca,'quasi_samps_unif.png','Resolution',600) 

% uniform random
scatter(2*rand(500,1)-1,2*rand(500,1)-1,dotsize,'k.')
exportgraphics(gca,'unif_samps.png','Resolution',600) 

% leverage random
indlev = randsample(N,500,true,levs);
[urows,ucols] = ind2sub([length(wvals) length(kvals)],indlev);
scatter(wvals(urows)-1,kvals(ucols)-2,dotsize,'b.')
exportgraphics(gca,'lev_samps.png','Resolution',600) 


        
