function [flag] = isS(p)

ll = linefun(p);
%disp(ll)
%disp(sum(sum(ll*ll')))
%disp(sum(sum(abs(ll*ll'))))
flag = sum(sum(ll*ll')) ~= sum(sum(abs(ll*ll')));
%disp(flag)





end

function [ll] = linefun(p)

num = size(p,1);

ll = zeros(num-2,1);
for i=1:(num-2)
    ll(i) = (p(i,2)-p(num,2))/(p(1,2)-p(num,2)) - (p(i,1)-p(num,1))/(p(1,1)-p(num,1));
end

end