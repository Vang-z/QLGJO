function [x, fitness, Convergence, pop] = IGJO(level, N, dim, lb, ub, Epochs, x, fitness, prob, f_func)

%% Initialize parameters
    % the OBL strategy
    x_obl = sort(fix(lb + ub - x), 2);
    fitness_obl = feval(f_func, N, level, x_obl, prob);
    fitness = [fitness; fitness_obl];
    x = [x; x_obl];
    [fitness, index] = sort(fitness, 'desc');
    fitness = fitness(1:N, :);
    x = x(index(1:N), :);

    Male_jackal = zeros(1, dim);
    Male_jackal_fit = -inf;

    Female_jackal = zeros(1, dim);
    Female_jackal_fit = -inf;

    Convergence = zeros(1, Epochs);
    pop = [];

%% Iteration
    for epoch = 1:Epochs
        rl = 0.05 * levy(N, dim, 1.5);
        E1 = 1.5 * (1 - (epoch / Epochs));

        % Choose the Male and Female Jackal
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
                E0 = 2 * rand - 1;
                E = E1 * E0;

                if abs(E) > 1
                    D_male_jackal = abs((Male_jackal(j) - rl(i, j) * x(i, j)));
                    Y1(j) = Male_jackal(j) - E * D_male_jackal;
                    D_female_jackal = abs((Female_jackal(j) - rl(i, j) * x(i, j)));
                    Y2(j) = Female_jackal(j) - E * D_female_jackal;
                else
                    D_male_jackal = abs((rl(i, j) * Male_jackal(j) - x(i, j)));
                    Y1(j) = Male_jackal(j) - E * D_male_jackal;
                    D_female_jackal = abs((rl(i, j) * Female_jackal(j) - x(i, j)));
                    Y2(j) = Female_jackal(j) - E * D_female_jackal;
                end
            end
            Y = fix((Y1 + Y2) / 2);
            Flag4lb = Y < lb;
            Flag4ub = Y > ub;
            Y = sort(Y .* (~(Flag4ub + Flag4lb)) + ub .* Flag4ub + lb .* Flag4lb);

            fit_y = feval(f_func, 1, level, Y, prob);
            if fit_y > fitness(i)
                x(i, :) = Y;
                fitness(i) = fit_y;
            end
        end

        x_obl = sort(fix(lb + ub - x), 2);
        fitness_obl = feval(f_func, N, level, x_obl, prob);
        fitness = [fitness; fitness_obl];
        x = [x; x_obl];
        [fitness, index] = sort(fitness, 'desc');
        fitness = fitness(1:N, :);
        x = x(index(1:N), :);

        Convergence(epoch) = max(fitness);
        pop = [pop; {x}];
    end
end
