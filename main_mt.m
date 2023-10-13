%% Add path
addpath(genpath('Multi-Thresholding/'));
addpath(genpath('Utils/'));

%% Initialize params
clear;
clc;
% Choose the fitness functions: otsu | kapur | tsallis
f_func = 'otsu';
levels = [8, 12, 16, 20];
images = {'Patient_3.png', 'Patient_4.png', 'Patient_5.png', 'Patient_6.png', 'Patient_7.png', 'Patient_9.png', 'Patient_13.png', 'Patient_24.png', 'Patient_121.png'};
population_size = 60;
max_iteration = 200;
N_algorithm = 1;
runtimes = 1;

for i_img = 1:9
img = cell2mat(images(i_img));
n = strsplit(img, '.');
for i_level = 1:length(levels)
level = levels(i_level);
Iin = imread(img);
set(0, 'defaultfigurecolor', 'w');
base = 0;
%hist2d(Iin)
for i_algorithm = 1:N_algorithm

switch i_algorithm
    case 1
        algorithm = 'QLGJO';
end

for i = 1:runtimes
    while 1
    try
    T = clock;
    fprintf('========================== [%s-%s-%s %s:%s:%s]: %s Running, runtimes: %d ==========================\n', ...
        num2str(T(1)), num2str(T(2)), num2str(T(3)), num2str(T(4)), num2str(T(5)), num2str(floor(T(6))), algorithm, i);
    tic;
    [intensity, Iout, prob, Convergence, pop] = MT(Iin, level, population_size, max_iteration, algorithm, f_func);
    runtime = toc;
    fprintf('Run time: %12fs\n', runtime);
    psnr4mt = psnr(Iin, Iout);
    ssim4mt = ssim(Iin, Iout);
    [fsim, fsim_c] = FSIM(Iin, Iout);
    Indicator.([cell2mat(n(1)), '_', num2str(level)]).(algorithm)(i).intensity = intensity;
    Indicator.([cell2mat(n(1)), '_', num2str(level)]).(algorithm)(i).Iout = Iout;
    Indicator.([cell2mat(n(1)), '_', num2str(level)]).(algorithm)(i).Convergence = Convergence;
    Indicator.([cell2mat(n(1)), '_', num2str(level)]).(algorithm)(i).pop = pop;
    Indicator.([cell2mat(n(1)), '_', num2str(level)]).(algorithm)(i).Fitness = mean(Convergence(:, end));
    Indicator.([cell2mat(n(1)), '_', num2str(level)]).(algorithm)(i).PSNR = psnr4mt;
    Indicator.([cell2mat(n(1)), '_', num2str(level)]).(algorithm)(i).SSIM = ssim4mt;
    Indicator.([cell2mat(n(1)), '_', num2str(level)]).(algorithm)(i).FSIM = fsim;
    Indicator.([cell2mat(n(1)), '_', num2str(level)]).(algorithm)(i).FSIMc = fsim_c;
    Indicator.([cell2mat(n(1)), '_', num2str(level)]).(algorithm)(i).runtime = runtime;

    disp('The best fitness: ');
    disp(['    Red:   ', num2str(Convergence(1, end))]);
    disp(['    Green: ', num2str(Convergence(2, end))]);
    disp(['    Blue:  ', num2str(Convergence(3, end))]);
    disp('The intensity: ');
    disp(['    Red:   ', num2str(intensity(1, :))]);
    disp(['    Green: ', num2str(intensity(2, :))]);
    disp(['    Blue:  ', num2str(intensity(3, :))]);
    disp(['The PSNR: ', num2str(psnr4mt)]);
    disp(['The SSIM: ', num2str(ssim4mt)]);
    disp(['The FSIM: ', num2str(fsim)]);
    disp(['The FSIMc: ', num2str(fsim_c)]);
    break;
    catch
    end
    end
end

%% Choose one result with the median indicators
choosen_In = [Indicator.([cell2mat(n(1)), '_', num2str(level)]).(algorithm).PSNR];
if i_algorithm == 1
choosen_In = rmmissing(choosen_In);
if mod(length(choosen_In), 2) == 2
choosen_In = choosen_In(~ismember(choosen_In, min(choosen_In)));
end
median_index = find(choosen_In == max(choosen_In));
base = Indicator.([cell2mat(n(1)), '_', num2str(level)]).(algorithm)(median_index(1)).PSNR;
else
choosen_In = rmmissing(choosen_In);
if mod(length(choosen_In), 2) == 2
choosen_In = choosen_In(~ismember(choosen_In, max(choosen_In)));
end
median_index = find(choosen_In == min(choosen_In));
end
A = Indicator.([cell2mat(n(1)), '_', num2str(level)]).(algorithm)(median_index(1));
A.Algorithm = algorithm;
A.PSNR_STD = std([Indicator.([cell2mat(n(1)), '_', num2str(level)]).(algorithm).PSNR]);
A.SSIM_STD = std([Indicator.([cell2mat(n(1)), '_', num2str(level)]).(algorithm).SSIM]);
A.FSIM_STD = std([Indicator.([cell2mat(n(1)), '_', num2str(level)]).(algorithm).FSIM]);
A.Fitness_STD = std([Indicator.([cell2mat(n(1)), '_', num2str(level)]).(algorithm).Fitness]')';
choose.([cell2mat(n(1)), '_', num2str(level)])(i_algorithm) = A;
if i_level == 1
Table.('PSNR')((i_img - 1) * 8 + 1, i_algorithm) = mean(A.PSNR);
Table.('PSNR')((i_img - 1) * 8 + 2, i_algorithm) = A.PSNR_STD;
Table.('SSIM')((i_img - 1) * 8 + 1, i_algorithm) = mean(A.SSIM);
Table.('SSIM')((i_img - 1) * 8 + 2, i_algorithm) = A.SSIM_STD;
Table.('FSIM')((i_img - 1) * 8 + 1, i_algorithm) = mean(A.FSIM);
Table.('FSIM')((i_img - 1) * 8 + 2, i_algorithm) = A.FSIM_STD;
elseif i_level == 2
Table.('PSNR')((i_img - 1) * 8 + 3, i_algorithm) = mean(A.PSNR);
Table.('PSNR')((i_img - 1) * 8 + 4, i_algorithm) = A.PSNR_STD;
Table.('SSIM')((i_img - 1) * 8 + 3, i_algorithm) = mean(A.SSIM);
Table.('SSIM')((i_img - 1) * 8 + 4, i_algorithm) = A.SSIM_STD;
Table.('FSIM')((i_img - 1) * 8 + 3, i_algorithm) = mean(A.FSIM);
Table.('FSIM')((i_img - 1) * 8 + 4, i_algorithm) = A.FSIM_STD;
elseif i_level == 3
Table.('PSNR')((i_img - 1) * 8 + 5, i_algorithm) = mean(A.PSNR);
Table.('PSNR')((i_img - 1) * 8 + 6, i_algorithm) = A.PSNR_STD;
Table.('SSIM')((i_img - 1) * 8 + 5, i_algorithm) = mean(A.SSIM);
Table.('SSIM')((i_img - 1) * 8 + 6, i_algorithm) = A.SSIM_STD;
Table.('FSIM')((i_img - 1) * 8 + 5, i_algorithm) = mean(A.FSIM);
Table.('FSIM')((i_img - 1) * 8 + 6, i_algorithm) = A.FSIM_STD;
else
Table.('PSNR')((i_img - 1) * 8 + 7, i_algorithm) = mean(A.PSNR);
Table.('PSNR')((i_img - 1) * 8 + 8, i_algorithm) = A.PSNR_STD;
Table.('SSIM')((i_img - 1) * 8 + 7, i_algorithm) = mean(A.SSIM);
Table.('SSIM')((i_img - 1) * 8 + 8, i_algorithm) = A.SSIM_STD;
Table.('FSIM')((i_img - 1) * 8 + 7, i_algorithm) = mean(A.FSIM);
Table.('FSIM')((i_img - 1) * 8 + 8, i_algorithm) = A.FSIM_STD;
end
clear A;
%if i_algorithm == 1
%figure();
%mesh3('channel', Iout(:, :, 1), 'theme', 'red');
%title('The red channel of the segmentation');
%figure();
%mesh3('channel', Iout(:, :, 2), 'theme', 'green');
%title('The green channel of the segmentation');
%figure();
%mesh3('channel', Iout(:, :, 3), 'theme', 'blue');
%title('The blue channel of the segmentation');
%end
end
end
end

save('MT_Indicator');
