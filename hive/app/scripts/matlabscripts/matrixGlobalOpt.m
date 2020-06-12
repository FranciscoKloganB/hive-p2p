function [ToptValue, ToptValueMR] = matrixGlobalOpt(A, v_, n)
    clearvars;
    close all;

    Topt = sdpvar(n,n,'full');
    F = [Topt(:) >= 0, ...
        Topt * ones(n,1) == ones(n,1), ...
        Topt.*(ones(n)-A) == 0, ...
        v_*Topt == v_];
    S = sdpsettings('solver', 'BMIBNB');
    optimize(F, norm(Topt-1/n*ones(n),2), S);
    
    ToptValue = value(Topt);
    ToptValueMR = max(abs(eig(Msdp-1/n*ones(n))));
end