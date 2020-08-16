clearvars;
close all;
clc;

samples = 512;
network_sizes = {8, 12, 16, 24, 32};
function_labels = {'mh', 'sdp_mh', 'mgo'};

results = containers.Map('KeyType','char', 'ValueType','any');

for i = 1:length(network_sizes)
    size = network_sizes{i};
    size_results = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for j = 1:length(function_labels)
        label = function_labels{j};
        size_results(label) = [];
    end

    for j = 1:samples
        % Create random symmetric topology, do not enforce connectivity.
        m = round(rand(size));
        sym_topology = double(logical(m + m' + eye(size)));
        % Create random steady-state vector for the topology.
        v_ = rand(1, size);
        v_ = v_./sum(v_);
        % Test all functions in functions array.
        for k = 1:length(function_labels)
            % Mixing rate default value is 1.0. Meaning that when infeasible
            % or other errors occur we can still log a 'default' value that
            % is slow.
            label = function_labels{k};
            mixing_rate = 1.0;
            switch label
                case 'mh'
                    Msrw = sym_topology ./ sum(sym_topology);
                    Mh = mhMod(Msrw, v_);

                    eigenvalues = eig(Mh - 1 / size * ones(size));
                    mixing_rate = max(abs(eigenvalues));
                case 'sdp_mh'
                    Mopt = sdpvar(size);
                    t = sdpvar;
                    I = eye(size);
                    constraints = [
                        Mopt(:) >= 0, ...
                        Mopt * ones(size, 1) == ones(size, 1), ...
                        Mopt .* (ones(size) - sym_topology) == 0, ...
                        -t*I <= (Mopt - 1 / size * ones(size)) <= t*I
                    ];
                    S = sdpsettings('solver', 'sedumi', 'verbose', 0, 'debug', 0);
                    result = optimize(constraints, t, S);
                    Mhopt = value(Mopt);
                    Mhopt = mhMod(Mhopt, v_);

                    eigenvalues = eig(Mhopt - 1 / size * ones(size));
                    mixing_rate = max(abs(eigenvalues));
                case 'mgo'
                    Msdp = sdpvar(size, size, 'full');
                    s = sdpvar;
                    I = eye(size);

                    constraints = [
                        Msdp(:) >= 0, ...
                        Msdp * ones(size, 1) == ones(size, 1), ...
                        Msdp .* (ones(size) - sym_topology) == 0, ...
                        v_ * Msdp == v_
                    ];

                    S = sdpsettings('solver', 'bmibnb', 'verbose', 0, 'debug', 0);
                    t = norm(Msdp - 1 / size * ones(size), 2);
                    result = optimize(constraints, t, S);
                    Msdp = value(Msdp);

                    eigenvalues = eig(Msdp - 1 / size * ones(size));
                    mixing_rate = max(abs(eigenvalues));
                otherwise
                    continue;
            end
            previous_results = size_results(label);
            updated_result = [previous_results, mixing_rate];
            size_results(label) = updated_result;
        end
    end
    results(num2str(size)) = size_results;
end

json_string = jsonencode(results);
fid = fopen('sample-matlab.json', 'w');
if fid == -1
    error('Cannot create JSON file');
end
fwrite(fid, json_string);
fclose(fid);
