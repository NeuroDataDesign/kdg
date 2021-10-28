%% Single Contamination Trial

d = 2;              % dimensions
n = 220;            % total number of points in sample
grid_density = 200; % number of points in each direction on grid
p = 0.09;           % contamination proportion

% Run a single trial of the contamination experiment
h = contamination_trial(n, grid_density, p, d, true)


%% Run Experiment with varying sample size
sample_size = logspace(2, 4, 10);
reps = 10;
grid_density = 200;
p = 0.09;
d = 2;
debug=false;
reps_list = [];
sample_list = [];
hellinger_dist_rkde = [];

for sample = sample_size
    disp(['Doing sample ', num2str(int32(sample))]) 
    for ii = 1:reps
        h = contamination_trial(int32(sample), grid_density, p, d, debug);
        disp(h);
        hellinger_dist_rkde = [hellinger_dist_rkde h];
        reps_list = [reps_list ii];
        sample_list = [sample_list int32(sample)];
    end
end

% Plotting
hellinger_rkde_med = [];
hellinger_rkde_25_quantile = [];
hellinger_rkde_75_quantile = [];

for sample = sample_size
   indices = sample_list == int32(sample);
   curr_hellinger = hellinger_dist_rkde(indices);
   hellinger_rkde_25_quantile(end + 1) = quantile(curr_hellinger, 0.25);
   hellinger_rkde_75_quantile(end + 1) = quantile(curr_hellinger, 0.75);
   hellinger_rkde_med(end + 1) = median(curr_hellinger);
end

% Plotting
figure;
hold on
plot(sample_size, hellinger_rkde_med, 'r')
in_between = [hellinger_rkde_25_quantile, fliplr(hellinger_rkde_75_quantile)];
x_ax = [sample_size, fliplr(sample_size)];
fill(x_ax, in_between, 'r', 'FaceAlpha', 0.3, 'LineStyle', 'none');
hold off

function [] = replicate_figure(X0, Xc, grid_density, true_pdf)
% Generate the 2D figure from the RKDE paper
% X0: the sampled points
% Xc: the contaminated points
% grid_density: dimension of grid in each direction
% true_pdf: the actual pdf of the Guassians
x = linspace(-6, 6, grid_density);
y = linspace(-6, 6, grid_density);
[xx, yy] = meshgrid(x, y);

figure();
hold on
scatter(X0(:, 1), X0(:, 2), 'k');
scatter(Xc(:, 1), Xc(:, 2), 'x', 'r')
xlim([-6, 6])
ylim([-6, 6])
pbaspect([1 1 1])
z_max = max(true_pdf, [], 'all');
levels = [z_max / 16, z_max / 8, z_max / 4, z_max / 2, z_max * 3 / 4];
contour(xx, yy, true_pdf, levels)
hold off
end


function [X0, Xc, true_pdf] = generate_distribution(points, grid_density, p, debug)
% Generate distribution from RKDE paper
% points: number of total sample points
% grid_density: number of points in each direction on square grid
% p: proportion of points that are contaminated

centers = [0, -3; 0, 3];
num_contamination = int32(points * p);
num_samples = points - num_contamination;
samples_per_blob = floor(num_samples / 2);

% Generate contamination
X0_x = [randn(samples_per_blob, 1)-3; randn(samples_per_blob, 1)+3];
X0_y = [randn(samples_per_blob, 1); randn(samples_per_blob, 1)];
X0 = [X0_x X0_y];
Xc = (6--6)*rand(num_contamination, 2) - 6;  % contamination

% Generate true pdf
x = linspace(-6, 6, grid_density);
y = linspace(-6, 6, grid_density);
[xx, yy] = meshgrid(x, y);
pos = [xx(:) yy(:)];
pos1 = pos;
pos2 = pos;
pos1(:, 1) =  pos1(:, 1) - 3;
pos2(:, 1) = pos2(:, 1) + 3;
mu = 0;
sigma = [1, 0; 0, 1];
rv1 = mvnpdf(pos1, mu, sigma);
rv2 = mvnpdf(pos2, mu, sigma);
rv1 = reshape(rv1, length(y), length(x));
rv2 = reshape(rv2, length(y), length(x));
true_pdf = rv1 + rv2;
% true_pdf = true_pdf / sum(sum(true_pdf));
true_pdf = true_pdf / 2;

% figure
% imagesc(true_pdf);
% colorbar
if debug
    replicate_figure(X0, Xc, grid_density, true_pdf)
end
end

function [h] = contamination_trial(points, grid_density, p, d, debug)
% Run a trial of the contamination experiment
% points: the number of input points
% grid_density: the number of points in each dimension for sampling truth
% p: proportion of contamination
% d: number of input feature dimensions
% debug: true/false flag for displaying figures

n = points;            % total number of points in sample
[X0, Xc, true_pdf] = generate_distribution(n, grid_density, p, debug);
X = [X0; Xc];

%b_type: bandwidth type: 1 -> lscv, 2 -> lkcv, 3 -> jakkola heuristic,
b_type = 1;
h = bandwidth_select(X, b_type);

%weights for RKDE
type = 2; %Hampel loss
[w_hm a b c] = robkde(X, h, type);

% Set up distance matrix
x = linspace(-6, 6, grid_density);
y = linspace(-6, 6, grid_density);
[xx, yy] = meshgrid(x, y);
y = [reshape(xx, [grid_density^2, 1]), reshape(yy, [grid_density^2, 1])];
Y = pdist2(X, y).^2';

% Rest of estimation
pd = gauss_kern(Y, h, d);
f_hm = pd*w_hm;

% compute hellinger
true_pdf_flattened = reshape(true_pdf, [grid_density^2, 1]);
h = hellinger(true_pdf_flattened, f_hm);

% Reshape into 2D distribtion
f_hm = reshape(f_hm, [grid_density, grid_density]);

if debug
   % Display estimated density
    figure;
    imagesc(f_hm);
    colorbar;
    title('Estimated Distribtion')

    % Display true density
    figure;
    imagesc(true_pdf)
    colorbar;
    title('True Distribution') 
end
end

function [h] = hellinger(p, q)
% Hellinger distance between two discrete distributions
h = sqrt(sum((sqrt(p) - sqrt(q)).^2, 'all')) / sqrt(2);
end
