function [ToptValue, ToptValueMR] = matrixGlobalOpt(A, v_)
    n = length(v_);
    U = ones(n) / n;
    
    Topt = sdpvar(n, n, 'full');
    F = [Topt(:) >= 0,...
        Topt * ones(n,1) == ones(n, 1),...
        Topt .* (ones(n)-A) == 0,...
        v_ * Topt==v_];
    S = sdpsettings('solver', 'bmibnb');

    diagnostics = optimize(F, norm(Topt-U, 2), S);

    if diagnostics.problem == 0
        sprintf('Feasible')
        ToptValue = value(Topt);
        ToptValueMR = max(abs(eig(Topt - U)));
        display(class(ToptValue))
        display(class(ToptValueMR))
        return
    elseif diagnostics.problem == 1
        sprintf('Unfeasible')
    else
        sprintf('Something else happened (ec: %d)', diagnostics.problem)
    end

    ToptValue = [];
    ToptValueMR = 1.0;
    display(class(ToptValue))
    display(class(ToptValueMR))
    return
end