%% ------------------------------------------------------------------------------
% Configuration File for Visual Odometry Algorithm
%% -------------------------------------------------------------------------------

% Path to the directories containing images
% data_params.path1 = '../data/FrameR_S/';
% data_params.path2 = '../data/FrameL_S/';
data_params.path1 = '../data/Frame_L/';
data_params.path2 = '../data/Frame_R/';
%load camera calibration datas
% load('../data/cam_params_060319.mat');
load('../data/cam_params_020519.mat');

% Use parallel threads (requires Parallel Processing Toolbox)
data_params.use_multithreads = 1;                % 0: disabled, 1: enabled


%% Parameters for Feature Extraction
vo_params.feature.nms_n = 6;                      % non-max-suppression: min. distance between maxima (in pixels)
vo_params.feature.nms_tau = 40;                   % non-max-suppression: interest point peakiness threshold
vo_params.feature.margin = 21;                    % leaving margin for safety while computing features ( >= 25)

%% Parameters for Feature Matching
vo_params.matcher.match_binsize = 50;             % matching bin width/height (affects efficiency only)
vo_params.matcher.match_radius = 400;             % matching radius (du/dv in pixels)
vo_params.matcher.match_disp_tolerance = 1;       % dx tolerance for stereo matches (in pixels)
vo_params.matcher.match_ncc_window = 21;          % window size of the patch for normalized cross-correlation
vo_params.matcher.match_ncc_tolerance = 0.6;      % threshold for normalized cross-correlation //
% !! TO-DO: add subpixel-refinement using parabolic fitting
vo_params.matcher.refinement = 2;                 % refinement (0=none,1=pixel,2=subpixel)

%% Paramters for Feature Selection using bucketing
vo_params.bucketing.max_features = 2;             % maximal number of features per bucket
vo_params.bucketing.bucket_width = 50;            % width of bucket
vo_params.bucketing.bucket_height = 50;           % height of bucket
% !! TO-DO: add feature selection based on feature tracking
vo_params.bucketing.age_threshold = 10;           % age threshold while feature selection

%% Paramters for motion estimation
% !! TO-DO: use Nister's algorithm for Rotation estimation (along with SLERP) and
% estimate translation using weighted optimization equation
vo_params.estim.ransac_iters = 1500;              % number of RANSAC iterations
vo_params.estim.inlier_threshold = 2.0;          % fundamental matrix inlier threshold
vo_params.estim.reweighing = 1;                  % lower border weights (more robust to calibration errors)
