clear all;

n = 5;
topology = round(rand(n));
sym_topology = double(logical(topology + topology' + eye(n)));
mu = (1:n)./sum(1:n);
matrixGlobalOpt(sym_topology, mu)

function Msdp = matrixGlobalOpt(sym_topology, mu)
    n = length(mu);

    Msdp = sdpvar(n,n,'full');
    positiveF = [Msdp(:) >= 0];
    stochasticF = [Msdp * ones(n,1) == ones(n,1)];
    networkF = [Msdp.*(ones(n)-sym_topology) == 0];
    vF = [mu*Msdp == mu];
    F = [positiveF, stochasticF, networkF, vF];
    S = sdpsettings('solver', 'BMIBNB');
    result = optimize(F, norm(Msdp-1/n*ones(n),2), S);

    Msdp = value(Msdp);
end