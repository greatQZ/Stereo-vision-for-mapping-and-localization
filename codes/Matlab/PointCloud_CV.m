% xyz1 = xlsread('xyz.xlsx');%%%%do it manuelly
% imLeft = imread('ImageL_new.jpg');
% imRight = imread('ImageR_new.jpg'); 
% 
% points3D = reshape(xyz, [600, 800, 3]);
% points3D = points3D ./ 1000;
% 
% ptCloud = pointCloud(points3D, 'Color', imLeft);
% 
% player3D = pcplayer([-20, 20], [-10, 10], [0, 50], 'VerticalAxis', 'y', ...
%     'VerticalAxisDir', 'down');
% 
% view(player3D, ptCloud);

files = dir(fullfile('C:\\Users\\m8avhru\\Documents\\Visual Studio 2012\\Projects\\ELAS_X64_FA _Mapping_020519\\ELAS x64\\ELAS-x64\\ELAS-x64\\','*.csv'));
% files = dir(fullfile('C:\\Users\\m8avhru\\Documents\\Visual Studio 2012\\Projects\\OpenCVTest_FA_Mapping_060319\OpenCVTest\\','*.csv'));
LengthFiles = length(files);
imagesL= dir(fullfile('C:\\Users\\m8avhru\\Documents\\Visual Studio 2012\\Projects\\FrameL_0205\\','*.jpg'));
map3D = robotics.OccupancyMap3D(5,'FreeThreshold', 0.1, 'OccupiedThreshold', 0.6, 'ProbabilitySaturation', [0.12 0.97]);
% posegraph3D = robotics.PoseGraph3D;
% informationmatrix = [1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1];
% for k = 1:LengthFiles
%     addRelativePose(posegraph3D, pose_66x7(k,:),informationmatrix);
% end
poses_m = [poses_global(:, 1:3)./1000 , poses_global(:, 4:7)];
% poses_m(:,[2,3])= poses_m(:,[3,2]);
for i= 1:20
        
    delimiter = '\t';
    formatSpec = '%f%f%f%[^\n\r]';
    fileID = fopen(strcat('C:\Users\m8avhru\Documents\Visual Studio 2012\Projects\ELAS_X64_FA _Mapping_020519\ELAS x64\ELAS-x64\ELAS-x64\',files(i).name),'r');
%     fileID = fopen(strcat('C:\Users\m8avhru\Documents\Visual Studio 2012\Projects\OpenCVTest_FA_Mapping_060319\OpenCVTest\',files(i).name),'r');
    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN,  'ReturnOnError', false);fclose(fileID);
    xyz = [dataArray{1:end-1}];
%     xyz(:,[2,3])= xyz(:,[3,2]);
    clearvars filename delimiter formatSpec fileID dataArray ans;
    imLeft = imread(strcat('C:\Users\m8avhru\Documents\Visual Studio 2012\Projects\FrameL_0205\',imagesL(i).name));
    points3D = reshape(xyz, [1200, 1600, 3]);
    points3D = points3D./1000;
    ptCloud = pointCloud(points3D, 'Color', imLeft);
    roi= [-2 2 -0.5 1 0 4];
    indices = findPointsInROI(ptCloud,roi);
    ptCloudB = select(ptCloud,indices);
%     pcname=[num2str(i), '.pcd'];
%     pcwrite(ptCloudB, pcname,'Encoding','ascii');
    poses = poses_m(i,:);
    insertPointCloud(map3D, poses, ptCloudB, 4);
    
end 
figure;
% inflate(map3D, 0.2)
show(map3D);
%     player3D = pcplayer([-5, 5], [-5, 5], [0, 5], 'VerticalAxis', 'y', ...
%     'VerticalAxisDir', 'down');
%     view(player3D, ptCloud38);