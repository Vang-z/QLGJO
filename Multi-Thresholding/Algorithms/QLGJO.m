function [x, fitness, Convergence, pop] = QLGJO(level, N, dim, lb, ub, Epochs, x, fitness, prob, f_func)
%% Initialize parameters
    Male_jackal = zeros(1, dim);
    Male_jackal_fit = -inf;

    Female_jackal = zeros(1, dim);
    Female_jackal_fit = -inf;

    % initialize the params of Q-Learning
    action_num = 3;
    Reward_table = zeros(action_num, action_num, N);
    Q_table = zeros(action_num, action_num, N);
    cur_state = randi(action_num);
    gamma = 0.5;
    lambda_initial = 0.9;
    lambda_final = 0.1;

    Convergence = zeros(1, Epochs);

    pop = [];

%% Iteration
    for epoch = 1:Epochs
        rl = 0.05 * levy(N, dim, 1.5);
        E1 = 1.5 * (1 - (epoch / Epochs));

        for i = 1:N
            if fitness(i) > Male_jackal_fit
                Male_jackal_fit = fitness(i);
                Male_jackal = x(i, :);
            elseif fitness(i) > Female_jackal_fit
                Female_jackal_fit = fitness(i);
                Female_jackal = x(i, :);
            end
        end

        for i = 1:N
            for j = 1:dim
                % r1 is a random number in [0, 1]
                r1 = rand();
                E0 = 2 * r1 - 1;
                % Evading energy
                E = E1 * E0;

                if (Q_table(cur_state, 1, i) >= Q_table(cur_state, 2, i) && Q_table(cur_state, 1, i) >= Q_table(cur_state, 3, i))
                    action = 1;
                    % Exploration
                    D_male_jackal = abs((Male_jackal(j) - rl(i, j) * x(i, j)));
                    Male_Positions(i, j) = Male_jackal(j) - E * D_male_jackal;
                    D_female_jackal = abs((Female_jackal(j) - rl(i,j) * x(i, j)));
                    Female_Positions(i, j) = Female_jackal(j) - E * D_female_jackal;
                elseif  (Q_table(cur_state, 2, i) >= Q_table(cur_state, 1, i) && Q_table(cur_state, 2, i) >= Q_table(cur_state, 3, i))
                    action = 2;
                    if rand > 0.5
                        D_male_jackal = abs((Male_jackal(j) - rl(i, j) * x(i, j)));
                        Male_Positions(i, j) = Male_jackal(j) - E * D_male_jackal;
                        D_female_jackal = abs((Female_jackal(j) - rl(i,j) * x(i, j)));
                        Female_Positions(i, j) = Female_jackal(j) - E * D_female_jackal;
                    else
                        D_male_jackal = abs((rl(i, j) * Male_jackal(j) - x(i, j)));
                        Male_Positions(i, j) = Male_jackal(j) - E * D_male_jackal;
                        D_female_jackal = abs((rl(i, j) * Female_jackal(j) - x(i, j)));
                        Female_Positions(i, j) = Female_jackal(j) - E * D_female_jackal;
                    end
                else
                    action = 3;
                    % Exploitation
                    D_male_jackal = abs((rl(i, j) * Male_jackal(j) - x(i, j)));
                    Male_Positions(i, j) = Male_jackal(j) - E * D_male_jackal;
                    D_female_jackal = abs((rl(i, j) * Female_jackal(j) - x(i, j)));
                    Female_Positions(i, j) = Female_jackal(j) - E * D_female_jackal;
                end
                Position(j) = fix(Male_Positions(i, j) + Female_Positions(i, j) / 2);
            end
            Flag4lb = Position < lb;
            Flag4ub = Position > ub;
            Position = sort(Position .* (~(Flag4ub + Flag4lb)) + ub .* Flag4ub + lb .* Flag4lb);
            new_fitness = feval(f_func, 1, level, Position, prob);

            if action == 1
                % exoplore mutation
                mutation = x(randi(N), :) + 0.85 .* (x(randi(N), :) - x(randi(N), :)) + 0.85 .* (x(randi(N), :) - x(randi(N), :));
                Flag4lb = mutation < lb;
                Flag4ub = mutation > ub;
                mutation = sort(fix(mutation .* (~(Flag4ub + Flag4lb)) + ub .* Flag4ub + lb .* Flag4lb));
                mutation_fit = feval(f_func, 1, level, mutation, prob);
                if mutation_fit > new_fitness
                    new_fitness = mutation_fit;
                    Position = mutation;
                end
            elseif action == 2
                % hybird mutation
                mutation = x(randi(N), :) + 0.85 .* (x(randi(N), :) - x(randi(N), :));
                Flag4lb = mutation < lb;
                Flag4ub = mutation > ub;
                mutation = sort(fix(mutation .* (~(Flag4ub + Flag4lb)) + ub .* Flag4ub + lb .* Flag4lb));
                mutation_fit = feval(f_func, 1, level, mutation, prob);
                if mutation_fit > new_fitness
                    new_fitness = mutation_fit;
                    Position = mutation;
                end
            else
                % exploitation mutation
                mutation = x(i, :) + 0.85 .* (x(randi(N), :) - x(randi(N), :));
                Flag4lb = mutation < lb;
                Flag4ub = mutation > ub;
                mutation = sort(fix(mutation .* (~(Flag4ub + Flag4lb)) + ub .* Flag4ub + lb .* Flag4lb));
                mutation_fit = feval(f_func, 1, level, mutation, prob);
                if mutation_fit > new_fitness
                    new_fitness = mutation_fit;
                    Position = mutation;
                end
            end

            if new_fitness > fitness(i)
                x(i, :) = Position;
                fitness(i) = new_fitness;
                Reward_table(cur_state, action, i) = 1;
            else
                Reward_table(cur_state, action, i) = -1;
            end

            % update the Q_table
            r =  Reward_table(cur_state, action, i);
            maxQ = max(Q_table(action, :, i));
            lambda = (lambda_initial + lambda_final) / 2 - (lambda_initial - lambda_final) / 2 * cos(pi * (1 - epoch / Epochs));
            Q_table(cur_state, action, i) = Q_table(cur_state, action, i) + lambda * (r + gamma * maxQ - Q_table(cur_state, action, i));
            cur_state = action;
        end

        Convergence(epoch) = max(fitness);
        pop = [pop; {x}];
    end
end
