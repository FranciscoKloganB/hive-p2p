function [Topt, MR] = matrixGlobalOpt(A, v_)
    n = length(v_);
    U = ones(n) / n;
    
    Topt = sdpvar(n, n, 'full');
    F = [Topt(:) >= 0,...
        Topt * ones(n,1) == ones(n, 1),...
        Topt .* (ones(n)-A) == 0,...
        v_ * Topt==v_];
    S = sdpsettings('solver', 'bmibnb', 'verbose', 0, 'debug', 0);
    diagnostics = optimize(F, norm(Topt-U, 2), S);

    if diagnostics.problem == 0
        sprintf('Feasible');
        Topt = value(Topt);
        MR = max(abs(eig(Topt - U)));
        return;
    elseif diagnostics.problem == 1
        sprintf('Unfeasible');
    else
        sprintf('Something else happened (ec: %d)', diagnostics.problem);
    end
    
    Topt = [];
    MR = 1337; % Python not capable of converting Inf to float('inf'), so we return a mixing rate will never be faster than MH w/o SDP optimization
    return;
end