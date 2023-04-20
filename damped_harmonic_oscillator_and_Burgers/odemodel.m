function [deriv] = odemodel(t,y,p)
    c = p(1);
    k = p(2);
    f = p(3);
    w = p(4);
	deriv = [y(2); f*cos(w*t) - k*y(1) - c*y(2)];
end
