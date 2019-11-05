i=1;
filename = 'voxel_p10_room.csv';
% fid = fopen(filename, 'wt');
for x = -3.45:0.1:7.45
    for y = -2.05:0.1:0.75 
        for z= -2.05:0.1:3.05
%             a(i,1) = x;
%             a(i,2) = y;
%             a(i,3) = z;
            a = [x y z];
            occval = getOccupancy(map3D,a);
            if(occval>0.7)
                occ_voxel(i,:)= [occval x y z];
                i=i+1;
            end
            if (x==7.45 && y==0.75 && z==3.05)
                break;
            end 
        end
    end
end
csvwrite(filename, occ_voxel);
% writematrix(occ_voxel,'voxel_p.csv', 'Delimiter', ';');

% for n = 1: i
%     occval = getOccupancy(map3D,a);
% end
