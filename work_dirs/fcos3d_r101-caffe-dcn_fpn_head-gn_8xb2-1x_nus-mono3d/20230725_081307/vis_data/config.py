dataset_type = 'NuScenesDataset'
data_root = '/home/efs/users/mateusz/data/nuscenes/'
class_names = [
    'car',
    'truck',
    'trailer',
    'bus',
    'construction_vehicle',
    'bicycle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'barrier',
]
metainfo = dict(classes=[
    'car',
    'truck',
    'trailer',
    'bus',
    'construction_vehicle',
    'bicycle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'barrier',
])
input_modality = dict(use_lidar=False, use_camera=True)
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=None),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='Resize3D', scale=(480, ), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img',
            'gt_bboxes',
            'gt_bboxes_labels',
            'attr_labels',
            'gt_bboxes_3d',
            'gt_labels_3d',
            'centers_2d',
            'depths',
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=None),
    dict(type='Resize3D', scale=(480, )),
    dict(type='Pack3DDetInputs', keys=[
        'img',
    ]),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='NuScenesDataset',
        data_root='/home/efs/users/mateusz/data/nuscenes/',
        data_prefix=dict(
            pts='',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        ann_file='nuscenes_infos_train.pkl',
        load_type='mv_image_based',
        pipeline=[
            dict(type='LoadImageFromFileMono3D', backend_args=None),
            dict(
                type='LoadAnnotations3D',
                with_bbox=True,
                with_label=True,
                with_attr_label=True,
                with_bbox_3d=True,
                with_label_3d=True,
                with_bbox_depth=True),
            dict(type='Resize3D', scale=(480, ), keep_ratio=True),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='Pack3DDetInputs',
                keys=[
                    'img',
                    'gt_bboxes',
                    'gt_bboxes_labels',
                    'attr_labels',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                    'centers_2d',
                    'depths',
                ]),
        ],
        metainfo=dict(classes=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
        ]),
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=False,
        box_type_3d='Camera',
        use_valid_flag=True,
        backend_args=None))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesDataset',
        data_root='/home/efs/users/mateusz/data/nuscenes/',
        data_prefix=dict(
            pts='',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        ann_file='nuscenes_infos_val.pkl',
        load_type='mv_image_based',
        pipeline=[
            dict(type='LoadImageFromFileMono3D', backend_args=None),
            dict(type='Resize3D', scale=(480, )),
            dict(type='Pack3DDetInputs', keys=[
                'img',
            ]),
        ],
        modality=dict(use_lidar=False, use_camera=True),
        metainfo=dict(classes=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
        ]),
        test_mode=True,
        box_type_3d='Camera',
        use_valid_flag=True,
        backend_args=None))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesDataset',
        data_root='/home/efs/users/mateusz/data/nuscenes/',
        data_prefix=dict(
            pts='',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        ann_file='nuscenes_infos_val.pkl',
        load_type='mv_image_based',
        pipeline=[
            dict(type='LoadImageFromFileMono3D', backend_args=None),
            dict(type='Resize3D', scale=(480, )),
            dict(type='Pack3DDetInputs', keys=[
                'img',
            ]),
        ],
        modality=dict(use_lidar=False, use_camera=True),
        metainfo=dict(classes=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
        ]),
        test_mode=True,
        box_type_3d='Camera',
        use_valid_flag=True,
        backend_args=None))
val_evaluator = dict(
    type='NuScenesMetric',
    data_root='/home/efs/users/mateusz/data/nuscenes/',
    ann_file='/home/efs/users/mateusz/data/nuscenes/nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=None)
test_evaluator = dict(
    type='NuScenesMetric',
    data_root='/home/efs/users/mateusz/data/nuscenes/',
    ann_file='/home/efs/users/mateusz/data/nuscenes/nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=None)
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer')
model = dict(
    type='FCOSMono3D',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            1.0,
            1.0,
            1.0,
        ],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='mmdet.ResNet',
        depth=101,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=None,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(
            False,
            False,
            True,
            True,
        )),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSMono3DHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        use_direction_classifier=True,
        diff_rad_by_sin=True,
        pred_attrs=True,
        pred_velo=True,
        dir_offset=0.7854,
        dir_limit_offset=0,
        strides=[
            8,
            16,
            32,
            64,
            128,
        ],
        group_reg_dims=(
            2,
            1,
            3,
            1,
            2,
        ),
        cls_branch=(256, ),
        reg_branch=(
            (256, ),
            (256, ),
            (256, ),
            (256, ),
            (),
        ),
        dir_branch=(256, ),
        attr_branch=(256, ),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=0.1111111111111111,
            loss_weight=1.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_attr=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        bbox_coder=dict(type='FCOS3DBBoxCoder', code_size=9),
        norm_on_bbox=True,
        centerness_on_reg=True,
        center_sampling=True,
        conv_bias=True,
        dcn_on_last_conv=True),
    train_cfg=dict(
        allowed_border=0,
        code_weight=[
            1.0,
            1.0,
            0.2,
            1.0,
            1.0,
            1.0,
            1.0,
            0.05,
            0.05,
        ],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.8,
        score_thr=0.05,
        min_bbox_size=0,
        max_per_img=200))
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=12, val_begin=3, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.3333333333333333,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[
            8,
            11,
        ],
        gamma=0.1),
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=35, norm_type=2))
auto_scale_lr = dict(enable=False, base_batch_size=16)
default_scope = 'mmdet3d'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
launcher = 'none'
work_dir = './work_dirs/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d'
