degree = 6;
step = .01;
% range to approximation leverage scores over
r = 1;
x = [-r:step:r];
true_levs = zeros(size(x));
upper_bound = (degree)./(2*sqrt(1-x.^2));

A = zeros(length(x),degree+1);
for i=0:degree
    A(:,i+1) = legendreP(i,x);
end

maxlines = zeros(length(x),length(x));
for i = 1:length(x)
    t = x(i);
    maxlines(:,i) = A(i,:)*pinv(A);
    maxlines(:,i) = (maxlines(:,i).^2)/(norm(maxlines(:,i))^2);
    true_levs(i) = maxlines(i,i);
end

%% visualize true leverage scores and upper bound
figure();
hold;
num_dark_plots = 1;
num_light_plots = 20;
% for i = 1:num_light_plots
%     plot(x,real(maxlines(:,randi(length(x)))),'Color',[0.4 0.6 1],'linewidth',1);
% end
for i = [1:8:201]
     plot(x,real(maxlines(:,i))/step,'Color',[0.4 0.6 1],'linewidth',1);
end
p(1) = plot(x,true_levs/step,'k','linewidth',2);
p(2) = plot(x,upper_bound, 'r-.', 'linewidth',2);
legend([p(1),p(2)],'leverage scores $\tau(\mu)$','approximation $\frac{q}{2\sqrt{1 - \mu)^2}}$', 'FontSize',18,'interpreter','latex','Location', 'north');
set(gca,'fontsize',16)
set(gca,'TickLabelInterpreter','latex');
xlabel('$\mu$','FontSize',20,'interpreter','latex');
xlim([-r,r])
ylim([0,20])

pbaspect([2 1 1])
exportgraphics(gca,'poly_levs.png','Resolution',600) 
