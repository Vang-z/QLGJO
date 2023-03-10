function mssim = SSIM(Iin, Iout)

[M N] = size(Iin);

window = fspecial('gaussian', 11, 1.5);
K(1) = 0.01;
K(2) = 0.03;
L = 255;

Iin = double(Iin);
Iout = double(Iout);

f = max(1, round(min(M, N) / 256));

%if f > 1
    lpf = ones(f, f);
    lpf = lpf / sum(lpf(:));
    Iin = imfilter(Iin, lpf, 'symmetric', 'same');
    Iout = imfilter(Iout, lpf, 'symmetric', 'same');

    Iin = Iin(1:f:end, 1:f:end);
    Iout = Iout(1:f:end, 1:f:end);
%end

C1 = (K(1) * L)^2;
C2 = (K(2) * L)^2;
window = window / sum(sum(window));

mu1 = filter2(window, Iin, 'valid');
mu2 = filter2(window, Iout, 'valid');
mu1_sq = mu1 .* mu1;
mu2_sq = mu2 .* mu2;
mu1_mu2 = mu1 .* mu2;

sigma1_sq = filter2(window, Iin .* Iin, 'valid') - mu1_sq;
sigma2_sq = filter2(window, Iout .* Iout, 'valid') - mu2_sq;
sigma12 = filter2(window, Iin .* Iout, 'valid') - mu1_mu2;

ssim_map = ((2*mu1_mu2 + C1) .* (2 * sigma12 + C2)) ./ ((mu1_sq + mu2_sq + C1) .* (sigma1_sq + sigma2_sq + C2));
mssim = mean2(ssim_map);
end
