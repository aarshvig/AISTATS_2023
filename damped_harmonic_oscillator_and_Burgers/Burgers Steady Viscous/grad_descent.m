function x = grad_descent(A,b,f,fprime,iter,eta)
% A = P2d(indunif,:);
% b = qoivec(indunif);
% f = @(x) max(x,0);
% fprime = @(x) (x>=0);
% iter = 10000;


% eta = .000005;
[n,d] = size(A);
x = pinv(A)*b;
errors = zeros(1,iter);
bnorm = sum(b.^2);
for i = 1:iter
    guess = A*x;
    error = sum((f(guess) - b).^2);
    errors(i) = error/bnorm;
    g = A'*diag(fprime(guess))*(f(guess)-b);
    x_new = x - eta*g;

    guess_new = A*x_new;
    error_new = sum((f(guess_new) - b).^2);
    if(error_new - error < .001*g'*(x_new - x))
          eta = eta*3;
          x = x_new;
    else
        eta = eta/(1+5/sqrt(i));
    end
end
% figure();
% plot(1:iter,errors)
end 