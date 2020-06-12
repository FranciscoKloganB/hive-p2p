function [ToptValue, ToptValueMR] = matrixGlobalOpt(A, v_)
    n = length(v_)
    U = ones(n) / n

    Topt = sdpvar(n, n, 'full');
    F = [Topt(:) >= 0, ...
        Topt * ones(n, 1) == ones(n, 1), ...
        Topt .* (ones(n) - A) == 0, ...
        v_ * Topt == v_];
    S = sdpsettings('solver', 'BMIBNB');
    optimize(F, norm(Topt - U, 2), S);

    ToptValue = value(Topt);
    display(ToptValue)

    ToptValueMR = max(abs(eig(Topt - U)));
    display(ToptValueMR)
end