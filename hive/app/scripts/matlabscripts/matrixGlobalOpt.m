function [ToptValue, ToptValueMR] = matrixGlobalOpt(A, v_)
    n = length(v_);
    U = ones(n) / n;
    
    Topt = sdpvar(n, n, 'full');
    F = [Topt(:) >= 0,...
        Topt * ones(n,1) == ones(n, 1),...
        Topt .* (ones(n)-A) == 0,...
        v_ * Topt==v_];
    S = sdpsettings('verbose', 0);
    diagnostics = optimize(F, norm(Topt-U, 2), S);

    if diagnostics.problem == 0
        sprintf('Feasible')
        ToptValue = cast(value(Topt), 'like', [0 1; 1 0]); % casts sdpvar type to double array (works - but python still doesn't recognize this...)
        ToptValueMR = max(abs(eig(Topt - U)));
        return
    elseif diagnostics.problem == 1
        sprintf('Unfeasible')
    else
        sprintf('Something else happened (ec: %d)', diagnostics.problem)
    end
    
    ToptValue = [];
    ToptValueMR = 1.0;
    return
end