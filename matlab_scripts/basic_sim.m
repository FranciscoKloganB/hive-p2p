% hive properties
worker_count = input('How many worker bees should be created? ' );

if worker_count == 0
    return;
end

desired_distr_vector = input(['Provide a desired distribution (column) vector  with size ' num2str(worker_count) '...']);

% simulation properties
simulation_stages = input('At most, how many stages should the simulation have? ');
multi_kill = input('At each discrete time stage, can more than a worker bee die? 0 for false, 1 for true; ');
kill_probability = input('Choose the likelihood ([0, 1]) that a worker will leave the network at each given stage. ');

% tracking structures
worker_health_tracker = ones(1, worker_count);

% file loading
file_id =  fopen('static/lorem_ipsum/honey.txt');
[file_bytes, file_size] = fread(file_id);
fclose(file_id);

% file split
disp(size(file_bytes))
display(getSha256(file_bytes))

function sha256val = getSha256(bytes)
    try
        enc = detect_encoding(bytes);
        string = native2unicode(bytes, enc);
        disp(string);
        hasher = System.Security.Cryptography.SHA256Managed;
        sha256 = uint8(hasher.ComputeHash(uint8(string)));
        sha256val = dec2hex(sha256);
    catch ME
        rethrow(ME);
    end
end