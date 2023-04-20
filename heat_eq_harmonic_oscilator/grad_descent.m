function x = grad_descent(A,b,weight_vec,f,fprime,iter,eta)
% A = P2d(indunif,:);
% b = qoivec(indunif);
% iter = 10000;


% eta = .000005;
[n,d] = size(A);
%x = pinv(A)*b;
x = zeros(d,1);
errors = zeros(1,iter);
bnorm = sum(weight_vec.*b.^2);
for i = 1:iter
    guess = A*x;
    error = sum(weight_vec.*(arrayfun(f,guess) - b).^2);
    errors(i) = error/bnorm;
    g = A'*diag(arrayfun(fprime,guess))*(weight_vec.*(arrayfun(f,guess)-b));
    x_new = x - eta*g;

    guess_new = A*x_new;
    error_new = sum(weight_vec.*(arrayfun(f,guess_new) - b).^2);
    if(error_new - error < .5*g'*(x_new - x))
        eta = eta*2;
        x = x_new;
    else
        eta = eta/2;
    end
end
% figure();
% semilogy(1:iter,errors)
end 