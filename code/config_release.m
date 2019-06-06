function param = config_release(id, iter_no, GPUdeviceNumber, caffepath)
%% set this part
if(nargin < 3)
    GPUdeviceNumber = 0;
end
if(nargin < 4)
    % !!!!!!!!  PLEASE SPECIFY PATH YOUR INSTALLED CAFFE PATH !!!!!!!!!
    caffepath = '/cluster/home/eli/multipeople3d/external/openpose/3rdparty/caffe/matlab';
end
param.use_gpu = 2;

if id == 1 % limb sampling
    param.scale_search = [1];
    param.thre1 = 0.1;
    param.thre2 = 0.05; 
    param.thre3 = 0.5; 
    param.thre5 = 0;
    param.model.caffemodel = sprintf('./data/model/LS_iter_%d.caffemodel', iter_no);
    param.model.deployFile = './data/model/deploy_sampling.prototxt';
    param.model.description = 'COCO - limb sampling trained on 3D, 2D and geom';
    param.model.boxsize = 368;
    param.model.padValue = 128;
    param.model.np = 18; 
    param.model.part_str = {'nose', 'neck', 'Rsho', 'Relb', 'Rwri', ... 
                             'Lsho', 'Lelb', 'Lwri', ...
                             'Rhip', 'Rkne', 'Rank', ...
                             'Lhip', 'Lkne', 'Lank', ...
                             'Leye', 'Reye', 'Lear', 'Rear', 'pt19'};
elseif id == 2 % limb scoring
    param.scale_search = [1];
    param.thre1 = 0.1;
    param.thre2 = 0.05; 
    param.thre3 = 0.5; 
    param.thre5 = 0;
    param.model.caffemodel = sprintf('./data/model/LS_iter_%d.caffemodel', iter_no);
    param.model.deployFile = './data/model/deploy_scoring.prototxt';
    param.model.description = 'COCO - limb scoring trained on 3D, 2D and geom';
    param.model.boxsize = 368;
    param.model.padValue = 128;
    param.model.np = 18; 
    param.model.part_str = {'nose', 'neck', 'Rsho', 'Relb', 'Rwri', ... 
                             'Lsho', 'Lelb', 'Lwri', ...
                             'Rhip', 'Rkne', 'Rank', ...
                             'Lhip', 'Lkne', 'Lank', ...
                             'Leye', 'Reye', 'Lear', 'Rear', 'pt19'};
end
disp(caffepath);
addpath(caffepath);
caffe.set_mode_gpu();
caffe.set_device(GPUdeviceNumber);
end

% 
% if id == 1 % COCO 2d - no vectors
%     param.scale_search = [1];
%     param.thre1 = 0.1;
%     param.thre2 = 0.05; 
%     param.thre3 = 0.5; 
%     param.thre5 = 0;
%     param.model.caffemodel = sprintf('/cluster/home/eli/multipeople3d/external/Realtime_Multi-Person_Pose_Estimation-master/testing/Release/2D_vecfree_original_loss/snapshots/_iter_%d.caffemodel', iter_no);
%     param.model.deployFile = '/cluster/home/eli/multipeople3d/external/Realtime_Multi-Person_Pose_Estimation-master/testing/Release/2D_vecfree_original_loss/models/deploy.prototxt';
%     param.model.description = 'COCO 2d - no vectors, individual loss';
%     param.model.boxsize = 368;
%     param.model.padValue = 128;
%     param.model.np = 18; 
%     param.model.part_str = {'nose', 'neck', 'Rsho', 'Relb', 'Rwri', ... 
%                              'Lsho', 'Lelb', 'Lwri', ...
%                              'Rhip', 'Rkne', 'Rank', ...
%                              'Lhip', 'Lkne', 'Lank', ...
%                              'Leye', 'Reye', 'Lear', 'Rear', 'pt19'};
% elseif id == 2 % COCO 2d - original PAF
%     param.scale_search = 1; %[0.5 1 1.5 2];
%     param.thre1 = 0.1;
%     param.thre2 = 0.05; 
%     param.thre3 = 0.5; 
%     param.thre5 = 0;
%     
%     param.model.caffemodel = '../model/_trained_COCO/pose_iter_440000.caffemodel';
%     param.model.deployFile = '../model/_trained_COCO/pose_deploy.prototxt';
%     param.model.description = 'COCO Pose56 Two-level Linevec';
%     param.model.boxsize = 368;
%     param.model.padValue = 128;
%     param.model.np = 18; 
%     param.model.part_str = {'nose', 'neck', 'Rsho', 'Relb', 'Rwri', ... 
%                              'Lsho', 'Lelb', 'Lwri', ...
%                              'Rhip', 'Rkne', 'Rank', ...
%                              'Lhip', 'Lkne', 'Lank', ...
%                              'Leye', 'Reye', 'Lear', 'Rear', 'pt19'};
% elseif id == 3 % H80k train 3d - no vectors
%     param.scale_search = [1];
%     param.thre1 = 0.1;
%     param.thre2 = 0.05; 
%     param.thre3 = 0.5; 
%     param.thre5 = 0;
%     param.model.caffemodel = sprintf('/cluster/home/eli/multipeople3d/external/Realtime_Multi-Person_Pose_Estimation-master/testing/Release/3D_vecfree/snapshots/3D__iter_%d.caffemodel', iter_no);
%     param.model.deployFile = '/cluster/home/eli/multipeople3d/external/Realtime_Multi-Person_Pose_Estimation-master/testing/Release/3D_vecfree/models/deploy.prototxt';
%     param.model.description = 'COCO 3d - no vectors, sum loss';
%     param.model.boxsize = 368;
%     param.model.padValue = 128;
%     param.model.np = 18; 
%     param.model.part_str = {'nose', 'neck', 'Rsho', 'Relb', 'Rwri', ... 
%                              'Lsho', 'Lelb', 'Lwri', ...
%                              'Rhip', 'Rkne', 'Rank', ...
%                              'Lhip', 'Lkne', 'Lank', ...
%                              'Leye', 'Reye', 'Lear', 'Rear', 'pt19'};
% elseif id == 4 % limb sampling
%     param.scale_search = [1];
%     param.thre1 = 0.1;
%     param.thre2 = 0.05; 
%     param.thre3 = 0.5; 
%     param.thre5 = 0;
%     param.model.caffemodel = sprintf('/cluster/home/eli/multipeople3d/external/Realtime_Multi-Person_Pose_Estimation-master/testing/Release/LimbScoring_vecfree/snapshots/LS_iter_%d.caffemodel', iter_no);
%     param.model.deployFile = '/cluster/home/eli/multipeople3d/external/Realtime_Multi-Person_Pose_Estimation-master/testing/Release/LimbScoring_vecfree/models/deploy_sampling.prototxt';
%     param.model.description = 'COCO - limb sampling trained on 3D, 2D and geom';
%     param.model.boxsize = 368;
%     param.model.padValue = 128;
%     param.model.np = 18; 
%     param.model.part_str = {'nose', 'neck', 'Rsho', 'Relb', 'Rwri', ... 
%                              'Lsho', 'Lelb', 'Lwri', ...
%                              'Rhip', 'Rkne', 'Rank', ...
%                              'Lhip', 'Lkne', 'Lank', ...
%                              'Leye', 'Reye', 'Lear', 'Rear', 'pt19'};
% elseif id == 5 % limb scoring
%     param.scale_search = [1];
%     param.thre1 = 0.1;
%     param.thre2 = 0.05; 
%     param.thre3 = 0.5; 
%     param.thre5 = 0;
%     param.model.caffemodel = sprintf('/cluster/home/eli/multipeople3d/external/Realtime_Multi-Person_Pose_Estimation-master/testing/Release/LimbScoring_vecfree/snapshots/LS_iter_%d.caffemodel', iter_no);
%     param.model.deployFile = '/cluster/home/eli/multipeople3d/external/Realtime_Multi-Person_Pose_Estimation-master/testing/Release/LimbScoring_vecfree/models/deploy_scoring.prototxt';
%     param.model.description = 'COCO - limb scoring trained on 3D, 2D and geom';
%     param.model.boxsize = 368;
%     param.model.padValue = 128;
%     param.model.np = 18; 
%     param.model.part_str = {'nose', 'neck', 'Rsho', 'Relb', 'Rwri', ... 
%                              'Lsho', 'Lelb', 'Lwri', ...
%                              'Rhip', 'Rkne', 'Rank', ...
%                              'Lhip', 'Lkne', 'Lank', ...
%                              'Leye', 'Reye', 'Lear', 'Rear', 'pt19'};
% end
