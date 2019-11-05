clc
clear 
%% Execute the configuration file to read parameters for data paths
addpath('config');
addpath(genpath('functions'));
configFile;

%% Starting parallel pooling (requires Parallel Processing Toolbox)
% This section takes a while to load for the first time
% To shutdown, run: delete(gcp('nocreate'));
if (isempty(gcp) && data_params.use_multithreads)
    parpool();
end

%% Read directories containing images
img_files1 = dir(strcat(data_params.path1,'*.jpg'));
img_files2 = dir(strcat(data_params.path2,'*.jpg'));

num_of_images = length(img_files1);

%% Read camera parameters
% K1 = stereoParams060319.CameraParameters1;
% K2 = stereoParams060319.CameraParameters2;
K1 = stereoParams020519.CameraParameters1;
K2 = stereoParams020519.CameraParameters2;
% Rotation = stereoParams060319.RotationOfCamera2;
% translation= stereoParams060319.TranslationOfCamera2;
Rotation = stereoParams020519.RotationOfCamera2;
translation= stereoParams020519.TranslationOfCamera2;
% translationVector = -translation * Rotation';
P1 = cameraMatrix(K1, eye(3), [0,0,0])';%%%%%%%%%%%%
P2 = cameraMatrix(K2, Rotation, translation)';

%% Initialize variables for odometry

Rpos = eye(3);
rpos = eye(3);

pos = [0;0;0];
poseToWorld = [Rpos, [0;0;0];[0,0,0],1];

estimated_Pose_world_prev = [Rpos,pos; [0,0,0],1];
% estimated_Pose_prev = [Pose., Pose.t'; [0,0,0],1];
% poses_global(1,:)= [pos', rotm2quat(Rpos)];
%initial pose graph 3D
posegraph3D = robotics.PoseGraph3D;

posegraph3D_world = robotics.PoseGraph3D;

informationmatrix = [1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1]; 
%% Start Algorithm
% Create an empty viewSet object to manage the data associated with each view.
vSet = viewSet;

%% Read images for time instant t
%     I2_l = undistortImage(rgb2gray(imread([img_files1(t).folder, '/', img_files1(t).name])), stereoParams060319.CameraParameters1);%%%%%%%%%%%%%%
%     I2_r = undistortImage(rgb2gray(imread([img_files2(t).folder, '/', img_files2(t).name])), stereoParams060319.CameraParameters2);%%%%%%%%%%%%%%
I2_l = undistortImage(rgb2gray(imread(strcat([img_files1(1).folder, '/', img_files1(1).name]))), K1);%%%%%%%%%%%%%%
I2_r = undistortImage(rgb2gray(imread(strcat([img_files2(1).folder, '/', img_files2(1).name]))), K2);%%%%%%%%%%%%%%
% I2_l = imread(strcat([img_files1(1).folder, '/', img_files1(1).name]));%%%%%%%%%%%%%%
% I2_r = imread(strcat([img_files2(1).folder, '/', img_files2(1).name]));%%%%%%%%%%%%%%
fprintf('Frame: %i\n', 1);

%% compute features
vo_previous.pts1_l = computeFeatures(I2_l, vo_params.feature);
vo_previous.pts1_r = computeFeatures(I2_r, vo_params.feature);
% Detect features. Increasing 'NumOctaves' helps detect large-scale
% features in high-resolution images. Use an ROI to eliminate spurious
% features around the edges of the image.
border = 50;
roi = [border, border, size(I2_l, 2)- 2*border, size(I2_l, 1)- 2*border];
prevPoints   = detectSURFFeatures(I2_l, 'NumOctaves', 4, 'NumScaleLevels', 6, 'ROI', roi);

% Extract features. Using 'Upright' features improves matching, as long as
% the camera motion involves little or no in-plane rotation.
prevFeatures = extractFeatures(I2_l, prevPoints, 'Upright', true);
% Add the first view. Place the camera associated with the first view
% and the origin, oriented along the Z-axis.
viewId = 1;
vSet = addView(vSet, viewId, 'Points', prevPoints, 'Orientation', ...
    eye(3), 'Location', [0,0,0]);
poseStereo = poses(vSet);
I1_l = I2_l;
I1_r = I2_r;
fprintf('\n---------------------------------\n');
%% Bootstraping for initialization
% start = 0;
for t =  2:num_of_images
    %% Implement SOFT for time instant t+1
    I2_l = undistortImage(rgb2gray(imread(strcat([img_files1(t).folder, '/', img_files1(t).name]))), K1);%%%%%%%%%%%%%%
    I2_r = undistortImage(rgb2gray(imread(strcat([img_files2(t).folder, '/', img_files2(t).name]))), K2);%%%%%%%%%%%%%%
    fprintf('Frame: %i\n', t);
%   stereo odometry
    [R, tr, vo_previous, bucketed_matches] = visualSOFT(t, I1_l, I2_l, I1_r, I2_r, P1, P2, vo_params, vo_previous);
    
    
    
    %% Estimated pose relative to global frame at t = 0

%     quat= tform2quat(tform);
%     trvec = tform2trvec(tform);
%     
    %intergration method 1
%     pos = pos +  (- R'* Rpos * tr');
%     Rpos = R' * Rpos;
    
    %integration method 2
    pos = pos + Rpos * tr';
    Rpos = R * Rpos;
   %integration method 3
    tform = [R', (-tr*R)'; [0,0,0], 1];
    poseToWorld = poseToWorld * tform';

    relativePose_World = tform;
    relativeQuat_world = tform2quat(relativePose_World);
    relativePose_World = [tform2trvec(relativePose_World),relativeQuat_world];
    addRelativePose(posegraph3D_world,relativePose_World,informationmatrix);
%     estimated_Pose_world_curr = [Rpos, pos; [0,0,0], 1];
%     relativePose_World = estimated_Pose_world_prev/estimated_Pose_world_curr;
%     relativeQuat_world = tform2quat(relativePose_World);
%     relativePose_World = [tform2trvec(relativePose_World),relativeQuat_world];
%     addRelativePose(posegraph3D_world,relativePose_World,informationmatrix);
%     estimated_Pose_world_prev =  estimated_Pose_world_curr ; 
    poseStereo.Orientation{t} = Rpos;
    poseStereo.Location{t} = pos';
    %% Detect, extract and match features for mono tracks.
%     if t==2
    currPoints   = detectSURFFeatures(I2_l, 'NumOctaves', 4, 'NumScaleLevels', 6, 'ROI', roi);
    currFeatures = extractFeatures(I2_l, currPoints, 'Upright', true);
    indexPairs = matchFeatures(prevFeatures, currFeatures, ...
            'MatchThreshold', 10, 'MaxRatio', .7, 'Unique',  true);

    matchedPoints1 = prevPoints(indexPairs(:, 1));
    matchedPoints2 = currPoints(indexPairs(:, 2));
    [relativeOrient, relativeLoc, inlierIdx] = helperEstimateRelativePose(...
        matchedPoints1, matchedPoints2, K1);
    
    viewId = t;
% Add the current view to the view set.
    vSet = addView(vSet, viewId, 'Points', currPoints);
% Store the point matches between the previous and the current views.
    vSet = addConnection(vSet, viewId-1, viewId, 'Matches', indexPairs(inlierIdx,:));
% Get the table containing the previous camera pose.
    prevPose = poses(vSet, viewId-1);
    prevOrientation = prevPose.Orientation{1};
    prevLocation    = prevPose.Location{1};
    
% Compute the current camera pose in the global coordinate system 
% relative to the first view.
    orientation = relativeOrient * prevOrientation;
% use location from stereo rereprojection
    location = prevLocation - tr * prevOrientation;
    vSet = updateView(vSet, viewId, 'Orientation', orientation, ...
        'Location', location);
    tracks = findTracks(vSet);
% Get camera poses for all views.
    camPoses = poses(vSet);    
%     Triangulate initial locations for the 3-D world points.
    xyzPoints = triangulateMultiview(tracks, camPoses, K1);
% Refine camera poses using bundle adjustment.
    [~, camPoses] = bundleAdjustment(xyzPoints, tracks, camPoses, ...
         K1, 'FixedViewIDs', 1, 'PointsUndistorted', true,  ...
         'MaxIterations', 1000);
        
    vSet = updateView(vSet, camPoses);% Update view set.
   
    

    
    prevPoints = currPoints; 
    prevFeatures = currFeatures;
    
    %% Prepare frames for next iteration
    I1_l = I2_l;
    I1_r = I2_r;

    %% Plot the odometry transformed data
    subplot(2, 2, [2, 4]);
    
    pos = [poseToWorld(4,1); poseToWorld(4,2); poseToWorld(4,3)];
    scatter( pos(1), pos(3), 'b', 'filled');
    title(sprintf('Odometry plot at frame %d', t))
    xlabel('x-axis (in mm)');
    ylabel('z-axis (in mm)');
    legend('Estimated Pose')
    hold on;

    %% Pause to visualize the plot
    pause(0.0001);
    fprintf('\n---------------------------------\n');
end
figure; 
show(posegraph3D_world);
% 
vSet = helperNormalizeViewSet(vSet, poseStereo);
%%Add a loop closure edge. Add this edge between two existing nodes from the current frame to a previous frame. Optimize the pose graph to adjust nodes based on the edge constraints and this loop closure. Store the optimized poses.
camPoses = poses(vSet);
estimatedPose_prev = [camPoses.Orientation{1},(camPoses.Location{1})'; [0,0,0], 1];
for k = 2: num_of_images
    estimatedPose_curr = [camPoses.Orientation{k},(camPoses.Location{k})'; [0,0,0], 1];
     % Relative pose between current and previous frame
    relativePose = estimatedPose_prev/estimatedPose_curr;
     % Relative orientation represented in quaternions
    relativeQuat = tform2quat(relativePose);
    % Relative pose as [x y z qw qx qy qz] 
    relativePose = [tform2trvec(relativePose),relativeQuat];
     % Add pose to pose graph
    addRelativePose(posegraph3D,relativePose,informationmatrix);
    estimatedPose_prev = estimatedPose_curr;
end
figure; 
show(posegraph3D);

loopedge = [eye(3), [0;0;0]; [0,0,0],1];
relativeQuat = tform2quat(loopedge);
relativePose = [tform2trvec(loopedge),relativeQuat];
loopcandidateframeid = 1;
currentframeid = 62;
addRelativePose(posegraph3D,relativePose,informationmatrix,...
           loopcandidateframeid,currentframeid);
          
optimizedPosegraph = optimizePoseGraph(posegraph3D);
 

%get the final poses 
optimizedposes = nodes(optimizedPosegraph);
figure; 
show(optimizedPosegraph);

addRelativePose(posegraph3D_world,relativePose,informationmatrix,...
           loopcandidateframeid,currentframeid);
optimizedPosegraph_world = optimizePoseGraph(posegraph3D_world);

optimizedposes_world = nodes(optimizedPosegraph_world);
figure; 
show(optimizedPosegraph_world);


          




