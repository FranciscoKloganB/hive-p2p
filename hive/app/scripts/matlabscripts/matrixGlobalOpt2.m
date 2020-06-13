clear vars;
n = 8;
topology = round(rand(n));
% disp(topology)
sym_topology = double(logical(topology + topology' + eye(n)));
% disp(sym_topology)
mu = (1:n)./sum(1:n);
% disp(mu)
matrixGlobalOpt(sym_topology, mu)

function isFeasible(diagnostics)
    sprintf('###########\nDIAGNOSTICS\n###########\n')
    disp(diagnostics)
    if diagnostics.problem == 0
     sprintf('Solver thinks it is feasible')
    elseif diagnostics.problem == 1
     sprintf('Solver thinks it is infeasible')
    else
     sprintf('Something else happened... error code: %d', diagnostics.problem)
    end
    sprintf('###########\nDIAGNOSTICS\n###########\n')
end

function Msdp = matrixGlobalOpt(sym_topology, mu)
    n = length(mu);

    Msdp = sdpvar(n,n,'full');
    
    positiveF = [Msdp(:) >= 0];
    stochasticF = [Msdp * ones(n,1) == ones(n,1)];
    networkF = [Msdp .*(ones(n)-sym_topology) == 0];
    vF = [mu*Msdp == mu];
    F = [positiveF, stochasticF, networkF, vF];
    S = sdpsettings('solver', 'swarm');
    
    diagnostics = optimize(F, norm(Msdp-1/n*ones(n),2), S);
    isFeasible(diagnostics);
    
    Msdp = value(Msdp);
end