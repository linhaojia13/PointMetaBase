## Train
# PointNeXt
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnext-xl.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnext-l.yaml wandb.use_wandb=True

CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnext-l.yaml wandb.use_wandb=False
CUDA_VISIBLE_DEVICES=2 bash script/main_segmentation.sh cfgs/s3dis/pointnext-b.yaml
CUDA_VISIBLE_DEVICES=2 bash script/main_segmentation.sh cfgs/s3dis/pointnext-s.yaml
CUDA_VISIBLE_DEVICES=2 bash script/main_segmentation.sh cfgs/s3dis/pointnext-l-gelu.yaml
CUDA_VISIBLE_DEVICES=3 bash script/main_segmentation.sh cfgs/s3dis/pointnext-l-gelu.yaml
CUDA_VISIBLE_DEVICES=3 bash script/main_segmentation.sh cfgs/s3dis/pointnext-l-geluln.yaml

# PointNeXtSep
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextsep-xl.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextsep-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnextsep-b.yaml wandb.use_wandb=True;CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnextsep-b.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextsep-b.yaml wandb.use_wandb=True;CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextsep-b.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextsepnope-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextseponepe-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextseponepeglu-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextseponepege-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextseponepegge-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextseponepehge-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextseponepexyzge-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextseponepeml-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnextsaseponepe-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnextsaseponepe-xl.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextsaseponepe-b.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextsaseponepe-s.yaml wandb.use_wandb=True


# PointMetaBaseline
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointmetabaselineepe-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointmetabaselineepedp-l.yaml wandb.use_wandb=True

CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointmetabaselinenope-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointmetabase-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointmetabase-xl.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointmetabase-b.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointmetabase-xxl.yaml wandb.use_wandb=True

CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointmetabase-xxxl.yaml wandb.use_wandb=True



# PointNeXtSepOneNormAct
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextsep_onenormact-l.yaml wandb.use_wandb=True

# PointNeXtRep
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextrep-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextrepcat128-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextrepcat256-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextrepcat256-xl.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=2 bash script/main_segmentation.sh cfgs/s3dis/pointnextrepcat256-b.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextrepcatall256-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnextrepcatall128-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextrep4cat256-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnextrep4cat256_64-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextrepmbcat256-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextrep3cat256_64-l.yaml wandb.use_wandb=True

# PointNeXtRepFPN
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnextrepuppathcat256-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextrepupdownpathcat256-l.yaml wandb.use_wandb=True

# PointNeXtRepTR
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextreptrcat256-l.yaml wandb.use_wandb=True

# PointNeXtRepFormer
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextrepformernoconv1cat256-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextrepformeraddskipcat256-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextrepformeraddskippostconv1cat256-l.yaml wandb.use_wandb=True


# PointNeXtRepLongRange
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextrep4cat256_64_dyngcn-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextrep3cat256_64_TR-l.yaml wandb.use_wandb=True


# PointNeXtRepSIM
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextrepsim-l.yaml wandb.use_wandb=True

# PointNeXtNBN
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnextnbn-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextnbn-l.yaml wandb.use_wandb=True

# ASSANet
CUDA_VISIBLE_DEVICES=3 bash script/main_segmentation.sh cfgs/s3dis/assanet.yaml wandb.use_wandb=False
CUDA_VISIBLE_DEVICES=2 bash script/main_segmentation.sh cfgs/s3dis/assanet.yaml
CUDA_VISIBLE_DEVICES=3 bash script/main_segmentation.sh cfgs/s3dis/assanet-l.yaml
CUDA_VISIBLE_DEVICES=2 bash script/main_segmentation.sh cfgs/s3dis/assanet-gelu.yaml
CUDA_VISIBLE_DEVICES=3 bash script/main_segmentation.sh cfgs/s3dis/assanet-ln.yaml

# PointNeXtAnis
CUDA_VISIBLE_DEVICES=2 bash script/main_segmentation.sh cfgs/s3dis/pointnext_anis-l.yaml
CUDA_VISIBLE_DEVICES=3 bash script/main_segmentation.sh cfgs/s3dis/pointnext_anis-b.yaml

# PointNeXtASSA
CUDA_VISIBLE_DEVICES=3 bash script/main_segmentation.sh cfgs/s3dis/pointnext_assa-b.yaml
CUDA_VISIBLE_DEVICES=3 bash script/main_segmentation.sh cfgs/s3dis/pointnext_assa-l.yaml
CUDA_VISIBLE_DEVICES=3 bash script/main_segmentation.sh cfgs/s3dis/pointnext_assa-xl.yaml

# PointNeXtASSAPart
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnext_assapart-l.yaml

# Profile Parameters, FLOPs, and Throughput
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnext-b.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnext-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=1 python examples/profile.py --cfg cfgs/s3dis/pointnext-xl.yaml batch_size=16 num_points=15000 timing=True
CUDA_VISIBLE_DEVICES=1 python examples/profile.py --cfg cfgs/s3dis/pointnext_assa-l.yaml batch_size=16 num_points=15000 timing=True
CUDA_VISIBLE_DEVICES=1 python examples/profile.py --cfg cfgs/s3dis/pointnext_assa-xl.yaml batch_size=16 num_points=15000 timing=True
CUDA_VISIBLE_DEVICES=1 python examples/profile.py --cfg cfgs/s3dis/pointnext_assapart-l.yaml batch_size=16 num_points=15000 timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextsep-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextsep_onenormact-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextsep-b.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrep-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrepcat128-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrepcat256-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrepsim-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextsep-xl.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrepcat256-xl.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrepcat256-b.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrepcatall256-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrepcatall128-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrep4cat256-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrep4cat256_64-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrep4cat256_64_dyngcn-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrep3cat256_64_TR-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrepmbcat256-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrep3cat256_64-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrepformernoconv1cat256-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrepformeraddskipcat256-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrepformeraddskippostconv1cat256-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrepuppathcat256-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextrepupdownpathcat256-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextreptrcat256-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextsepnope-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextseponepe-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextseponepeglu-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextseponepege-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextseponepegge-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextseponepehge-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextseponepexyzge-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextseponepeml-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextsaseponepe-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextsaseponepe-xl.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointnextsaseponepe-b.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=1 python examples/profile.py --cfg cfgs/s3dis/pointnextsaseponepe-s.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=1 python examples/profile.py --cfg cfgs/s3dis/pointmetabaselineepe-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointmetabaselinenope-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=1 python examples/profile.py --cfg cfgs/s3dis/pointmetabase-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointmetabase-xl.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointmetabase-b.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointmetabase-xxl.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointmetabase-xxxl.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointmetabaselineepedp-l.yaml batch_size=16 num_points=15000 flops=True timing=True


## Test
# PointNeXt-XL
CUDA_VISIBLE_DEVICES=2 bash script/main_segmentation.sh cfgs/s3dis/pointnext-xl.yaml wandb.use_wandb=False mode=test --pretrained_path log/s3dis/s3dis-train-pointnext-xl-ngpus1-seed147-20220822-133109-CKfZDHHCogLqPH39imGBvs/checkpoint/s3dis-train-pointnext-xl-ngpus1-seed147-20220822-133109-CKfZDHHCogLqPH39imGBvs_ckpt_best.pth
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnext-xl.yaml wandb.use_wandb=False mode=test --pretrained_path /userhome/zyx/dataset/pointcloud/pointnext/s3disfull-train-pointnext-XL-ngpus1-seed9042-batch_size=16-dataset.common.test_area=1-20220427-233111-iA4eBGUXrkmfyoko9eAXjv/checkpoint/s3disfull-train-pointnextXLC64k32-ngpus1-seed9042-batch_size=16-dataset.common.test_area=1-20220427-233111-iA4eBGUXrkmfyoko9eAXjv_ckpt_best.pth
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnext-xl.yaml wandb.use_wandb=False mode=test --pretrained_path /userhome/zyx/dataset/pointcloud/pointnext/s3disfull-train-pointnext-XL-ngpus1-seed4737-batch_size=16-dataset.common.test_area=2-20220427-233145-ELZgRXWhzH849rYpARE5EZ/checkpoint/s3disfull-train-pointnextXLC64k32-ngpus1-seed4737-batch_size=16-dataset.common.test_area=2-20220427-233145-ELZgRXWhzH849rYpARE5EZ_ckpt_best.pth
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnext-xl.yaml wandb.use_wandb=False mode=test --pretrained_path /userhome/zyx/dataset/pointcloud/pointnext/s3disfull-train-pointnext-XL-ngpus1-seed4332-batch_size=16-dataset.common.test_area=3-20220427-233149-TAnTGs6Be7Zwzn3nBdeHno/checkpoint/s3disfull-train-pointnextXLC64k32-ngpus1-seed4332-batch_size=16-dataset.common.test_area=3-20220427-233149-TAnTGs6Be7Zwzn3nBdeHno_ckpt_best.pth
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnext-xl.yaml wandb.use_wandb=False mode=test --pretrained_path /userhome/zyx/dataset/pointcloud/pointnext/s3disfull-train-pointnext-XL-ngpus1-seed8917-batch_size=16-dataset.common.test_area=4-20220427-233149-9S7vVPdnEnExL3aVsLTee7/checkpoint/s3disfull-train-pointnextXLC64k32-ngpus1-seed8917-batch_size=16-dataset.common.test_area=4-20220427-233149-9S7vVPdnEnExL3aVsLTee7_ckpt_best.pth
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnext-xl.yaml wandb.use_wandb=False mode=test --pretrained_path /userhome/zyx/dataset/pointcloud/pointnext/pointnext-xl-area5/checkpoint/pointnext-xl_ckpt_best.pth
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnext-xl.yaml wandb.use_wandb=False mode=test --pretrained_path /userhome/zyx/dataset/pointcloud/pointnext/s3disfull-train-pointnext-XL-ngpus1-seed3684-batch_size=16-dataset.common.test_area=6-20220427-233154-LLMeaboFimDUf4yTNxRJDU/checkpoint/s3disfull-train-pointnextXLC64k32-ngpus1-seed3684-batch_size=16-dataset.common.test_area=6-20220427-233154-LLMeaboFimDUf4yTNxRJDU_ckpt_best.pth
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnext-xl.yaml wandb.use_wandb=False mode=test --pretrained_path log/for_6fold_pointnext-xl/s3disfull-train-pointnext-XL-ngpus1-seed8917-batch_size=16-dataset.common.test_area=4-20220427-233149-9S7vVPdnEnExL3aVsLTee7/checkpoint/s3disfull-train-pointnextXLC64k32-ngpus1-seed8917-batch_size=16-dataset.common.test_area=4-20220427-233149-9S7vVPdnEnExL3aVsLTee7_ckpt_best.pth

CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointmetabase-l.yaml wandb.use_wandb=False mode=test --pretrained_path log/for_s3dis_area5_best/s3dis-train-pointmetabase-l-ngpus1-seed4333-20221107-205749-LoivDYbY7pVF6pJtNKokEU/checkpoint/s3dis-train-pointmetabase-l-ngpus1-seed4333-20221107-205749-LoivDYbY7pVF6pJtNKokEU_ckpt_best.pth
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointmetabase-xl.yaml wandb.use_wandb=False mode=test --pretrained_path log/for_s3dis_area5_best/s3dis-train-pointmetabase-xl-ngpus1-seed2425-20221107-095339-3Ebwo3FGeyLhVKzTjgM8EK/checkpoint/s3dis-train-pointmetabase-xl-ngpus1-seed2425-20221107-095339-3Ebwo3FGeyLhVKzTjgM8EK_ckpt_best.pth
CUDA_VISIBLE_DEVICES=2 bash script/main_segmentation.sh cfgs/s3dis/pointmetabase-xxl.yaml wandb.use_wandb=False mode=test --pretrained_path log/for_s3dis_area5_best/s3dis-train-pointmetabase-xxl-ngpus1-seed1111-20221108-092238-kkyyxcTLq7MLLkumjivREZ/checkpoint/s3dis-train-pointmetabase-xxl-ngpus1-seed1111-20221108-092238-kkyyxcTLq7MLLkumjivREZ_ckpt_best.pth

# PointNeXt-L
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnext-l.yaml wandb.use_wandb=False mode=test --pretrained_path log/s3dis/s3dis-train-pointnext-l-ngpus1-seed2425-20220923-172432-nHN8SvaGm6VkGmzjZRqxxh/checkpoint/s3dis-train-pointnext-l-ngpus1-seed2425-20220923-172432-nHN8SvaGm6VkGmzjZRqxxh_ckpt_best.pth
# PointNeXtSep-B
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnextsep-b.yaml wandb.use_wandb=False mode=test --pretrained_path log/s3dis/s3dis-train-pointnextsep-b-ngpus1-seed2425-20221001-003053-h5LpszZJLtUngLT5Hp8xYX/checkpoint/s3dis-train-pointnextsep-b-ngpus1-seed2425-20221001-003053-h5LpszZJLtUngLT5Hp8xYX_ckpt_best.pth
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextsep-b.yaml wandb.use_wandb=False mode=test --pretrained_path log/s3dis/s3dis-train-pointnextsep-b-ngpus1-seed2425-20221001-003053-HNvfJqxCtLx2g8Fn5CH2XK/checkpoint/s3dis-train-pointnextsep-b-ngpus1-seed2425-20221001-003053-HNvfJqxCtLx2g8Fn5CH2XK_ckpt_best.pth
# PointNeXtRepCat256-B
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnextsep-b.yaml wandb.use_wandb=False mode=test --pretrained_path log/s3dis/s3dis-train-pointnextsep-b-ngpus1-seed2425-20221001-003053-h5LpszZJLtUngLT5Hp8xYX/checkpoint/s3dis-train-pointnextsep-b-ngpus1-seed2425-20221001-003053-h5LpszZJLtUngLT5Hp8xYX_ckpt_best.pth
# PointNeXtSepOnePE-L
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnextseponepe-l.yaml wandb.use_wandb=False mode=test --pretrained_path log/s3dis/s3dis-train-pointnextseponepe-l-ngpus1-seed2425-20221023-211109-YyLxKjKKiJvmFeyQiYyEUV/checkpoint/s3dis-train-pointnextseponepe-l-ngpus1-seed2425-20221023-211109-YyLxKjKKiJvmFeyQiYyEUV_ckpt_best.pth
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextseponepe-l.yaml wandb.use_wandb=False mode=test --pretrained_path log/s3dis/s3dis-train-pointnextseponepe-l-ngpus1-seed4333-20221023-211504-JtNgYGdBhPyf2BJngRxnrg/checkpoint/s3dis-train-pointnextseponepe-l-ngpus1-seed4333-20221023-211504-JtNgYGdBhPyf2BJngRxnrg_ckpt_best.pth
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnextseponepegge-l.yaml wandb.use_wandb=False mode=test --pretrained_path log/s3dis/s3dis-train-pointnextseponepegge-l-ngpus1-seed2425-20221026-102901-X6MswA6JaoSqEs9exPUUxs/checkpoint/s3dis-train-pointnextseponepegge-l-ngpus1-seed2425-20221026-102901-X6MswA6JaoSqEs9exPUUxs_ckpt_best.pth
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnextseponepegge-l.yaml wandb.use_wandb=False mode=test --pretrained_path log/s3dis/s3dis-train-pointnextseponepegge-l-ngpus1-seed4333-20221026-102949-Zkm7Mo4Ld6o6nJ3Gtut4EK/checkpoint/s3dis-train-pointnextseponepegge-l-ngpus1-seed4333-20221026-102949-Zkm7Mo4Ld6o6nJ3Gtut4EK_ckpt_best.pth

## 6-fold Test
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/test_s3dis_6fold.py --cfg cfgs/s3dis/pointnextsep-l.yaml mode=test wandb.use_wandb=False --pretrained_path log/for_6fold_pointnextsep-l/
CUDA_VISIBLE_DEVICES=1 python examples/segmentation/test_s3dis_6fold.py --cfg cfgs/s3dis/pointnextseponepegge-l.yaml mode=test wandb.use_wandb=False --pretrained_path log/for_6fold_pointnextseponepegge-l/
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/test_s3dis_6fold.py --cfg cfgs/s3dis/pointnextseponepe-l.yaml mode=test wandb.use_wandb=False --pretrained_path log/for_6fold_pointnextseponepe-l/
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/test_s3dis_6fold.py --cfg cfgs/s3dis/pointnext-xl.yaml mode=test --pretrained_path log/for_6fold_pointnext-xl
CUDA_VISIBLE_DEVICES=1 python examples/segmentation/test_s3dis_6fold.py --cfg cfgs/s3dis/pointnext-l.yaml mode=test --pretrained_path log/for_6fold_pointnext-l
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/test_s3dis_6fold.py --cfg cfgs/s3dis/pointnextsaseponepe-l.yaml mode=test --pretrained_path log/for_6fold_pointnextsaseponepe-l
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/test_s3dis_6fold.py --cfg cfgs/s3dis/pointnextsaseponepe-xl.yaml mode=test --pretrained_path log/for_6fold_pointnextsaseponepe-xl
CUDA_VISIBLE_DEVICES=1 python examples/segmentation/test_s3dis_6fold.py --cfg cfgs/s3dis/pointnextsaseponepe-b.yaml mode=test --pretrained_path log/for_6fold_pointnextsaseponepe-b
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/test_s3dis_6fold.py --cfg cfgs/s3dis/pointmetabase-l.yaml mode=test --pretrained_path log/for_6fold_pointmetabase-l
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/test_s3dis_6fold.py --cfg cfgs/s3dis/pointmetabase-xl.yaml mode=test --pretrained_path log/for_6fold_pointmetabase-xl
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/test_s3dis_6fold.py --cfg cfgs/s3dis/pointmetabase-xxl.yaml mode=test --pretrained_path log/for_6fold_pointmetabase-xxl
CUDA_VISIBLE_DEVICES=1 python examples/segmentation/test_s3dis_6fold.py --cfg cfgs/s3dis/pointmetabase-xxl.yaml mode=test --pretrained_path log/for_6fold_pointmetabase-xxl-2

######
python -m pip install --upgrade pip setuptools wheel
pip uninstall setuptools
pip install setuptools==59.5.0




####################
####ScanObjectNN####
####################
## Train
#PointNext-S
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/pointnext-s.yaml wandb.use_wandb=True
#PointNextSASep-S
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/pointnextsasep-s.yaml wandb.use_wandb=True
#PointMetaBase-S
CUDA_VISIBLE_DEVICES=2 python examples/classification/main.py --cfg cfgs/scanobjectnn/pointmetabase-s.yaml wandb.use_wandb=True
#PointMetaBase-B
CUDA_VISIBLE_DEVICES=2 python examples/classification/main.py --cfg cfgs/scanobjectnn/pointmetabase-b.yaml wandb.use_wandb=True

#### Profile Parameters, FLOPs, and Throughput
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/scanobjectnn/pointnext-s.yaml batch_size=128 num_points=1024 timing=True flops=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/scanobjectnn/pointnextsasep-s.yaml batch_size=128 num_points=1024 timing=True flops=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/scanobjectnn/pointmetabase-s.yaml batch_size=128 num_points=1024 timing=True flops=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/scanobjectnn/pointnet.yaml batch_size=128 num_points=1024 timing=True flops=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/scanobjectnn/pointnet++.yaml batch_size=128 num_points=1024 timing=True flops=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/scanobjectnn/dgcnn.yaml batch_size=128 num_points=1024 timing=True flops=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/scanobjectnn/pointmlp.yaml batch_size=128 num_points=1024 timing=True flops=True

## Test
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/pointmetabase-s.yaml  mode=test --pretrained_path log/for_scanobjectnn_best/scanobjectnn-train-pointmetabase-s-ngpus1-seed4333-20221106-130123-MawjM7ebxP92scLj78Wt5z/checkpoint/scanobjectnn-train-pointmetabase-s-ngpus1-seed4333-20221106-130123-MawjM7ebxP92scLj78Wt5z_ckpt_best.pth


####################
#####ModelNet40#####
####################
## train
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnext-s_c64.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=1 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointmetabase-s.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=3 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointmetabase-s_c64.yaml wandb.use_wandb=True

#### Profile Parameters, FLOPs, and Throughput
CUDA_VISIBLE_DEVICES=2 python examples/profile.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml batch_size=128 num_points=1024 timing=True flops=True
CUDA_VISIBLE_DEVICES=2 python examples/profile.py --cfg cfgs/modelnet40ply2048/pointmetabase-s.yaml batch_size=128 num_points=1024 timing=True flops=True


####################
####ShapeNetPart####
####################
## Train
#PointNext-S
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointnext-s.yaml 
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointnext-s.yaml wandb.use_wandb=True 
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointnext-s_c64.yaml wandb.use_wandb=True 
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointnext-s_c160.yaml wandb.use_wandb=True 
#PointMetaBase-S
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointmetabase-s.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointmetabase-s_c64.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=4,5,6,7 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointmetabase-s_c64.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointmetabase-s_c160.yaml wandb.use_wandb=True

#### Profile Parameters, FLOPs, and Throughput
CUDA_VISIBLE_DEVICES=2 python examples/profile.py batch_size=64 num_points=2048 timing=True --cfg cfgs/shapenetpart/pointnext-s.yaml model.encoder_args.in_channels=3 model.encoder_args.width=32 
CUDA_VISIBLE_DEVICES=2 python examples/profile.py --cfg cfgs/shapenetpart/pointnext-s.yaml batch_size=64 num_points=2048 timing=True
CUDA_VISIBLE_DEVICES=2 python examples/profile.py --cfg cfgs/shapenetpart/pointnext-s_c64.yaml batch_size=64 num_points=2048 timing=True
CUDA_VISIBLE_DEVICES=2 python examples/profile.py --cfg cfgs/shapenetpart/pointnext-s_c160.yaml batch_size=64 num_points=2048 timing=True
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/shapenetpart/pointmetabase-s.yaml batch_size=64 num_points=2048 timing=True
CUDA_VISIBLE_DEVICES=2 python examples/profile.py --cfg cfgs/shapenetpart/pointmetabase-s_c64.yaml batch_size=64 num_points=2048 timing=True
CUDA_VISIBLE_DEVICES=2 python examples/profile.py --cfg cfgs/shapenetpart/pointmetabase-s_c160.yaml batch_size=64 num_points=2048 timing=True

## Test
CUDA_VISIBLE_DEVICES=0 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointmetabase-s.yaml mode=test wandb.use_wandb=False --pretrained_path log/for_shapenetpart_best/shapenetpart-train-pointmetabase-s-ngpus4-seed2425-20221116-062945-jeBt4safu8h6fLSPcF6Jgx/checkpoint/shapenetpart-train-pointmetabase-s-ngpus4-seed2425-20221116-062945-jeBt4safu8h6fLSPcF6Jgx_ckpt_best.pth
CUDA_VISIBLE_DEVICES=0 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointmetabase-s_c64.yaml mode=test wandb.use_wandb=False --pretrained_path log/for_shapenetpart_best/shapenetpart-train-pointmetabase-s_c64-ngpus4-seed2425-20221116-095644-muGBLW4UDdLLFJ5RnJwivh/checkpoint/shapenetpart-train-pointmetabase-s_c64-ngpus4-seed2425-20221116-095644-muGBLW4UDdLLFJ5RnJwivh_ckpt_best.pth
CUDA_VISIBLE_DEVICES=0 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointmetabase-s_c160.yaml mode=test wandb.use_wandb=False --pretrained_path log/for_shapenetpart_best/shapenetpart-train-pointmetabase-s_c160-ngpus4-seed2425-20221115-171932-o8PAiBYT29E5ARSHrk6425/checkpoint/shapenetpart-train-pointmetabase-s_c160-ngpus4-seed2425-20221115-171932-o8PAiBYT29E5ARSHrk6425_ckpt_best.pth






#####################
#######ScanNet#######
#####################
## Train
# PointNeXt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/segmentation/main.py --cfg cfgs/scannet/pointnext-xl.yaml 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/segmentation/main.py --cfg cfgs/scannet/pointnext-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/segmentation/main.py --cfg cfgs/scannet/pointmetabase-l.yaml wandb.use_wandb=True
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/segmentation/main.py --cfg cfgs/scannet/pointmetabase-b.yaml wandb.use_wandb=True

## Test
# PointNeXt
CUDA_VISIBLE_DEVICES=0  python examples/segmentation/main.py --cfg cfgs/scannet/pointnext-l.yaml mode=test dataset.test.split=val --pretrained_path log/scannet/scannet-train-pointnext-l-ngpus8-seed4333-20221113-125814-BQ77yDzbMRovBZQdoQkNUn/checkpoint/scannet-train-pointnext-l-ngpus8-seed4333-20221113-125814-BQ77yDzbMRovBZQdoQkNUn_ckpt_best.pth
CUDA_VISIBLE_DEVICES=1  python examples/segmentation/main.py --cfg cfgs/scannet/pointmetabase-b.yaml mode=test dataset.test.split=val --pretrained_path log/scannet/scannet-train-pointmetabase-b-ngpus8-seed4333-20221114-234449-nDPDEVykrrbhmYcCEHzFFQ/checkpoint/scannet-train-pointmetabase-b-ngpus8-seed4333-20221114-234449-nDPDEVykrrbhmYcCEHzFFQ_ckpt_best.pth

CUDA_VISIBLE_DEVICES=0  python examples/segmentation/main.py --cfg cfgs/scannet/pointmetabase-l.yaml mode=test dataset.test.split=val --pretrained_path log/scannet/scannet-train-pointmetabase-l-ngpus8-seed4333-20221114-111740-FprQZ3hvJd6mKyW86bEuXo/checkpoint/scannet-train-pointmetabase-l-ngpus8-seed4333-20221114-111740-FprQZ3hvJd6mKyW86bEuXo_ckpt_best.pth
CUDA_VISIBLE_DEVICES=1  python examples/segmentation/main.py --cfg cfgs/scannet/pointmetabase-xl.yaml mode=test dataset.test.split=val --pretrained_path log/scannet/scannet-train-pointmetabase-xl-ngpus8-seed8620-20221114-191249-EVPtXbhn5qpcyN3Mwfhjio/checkpoint/scannet-train-pointmetabase-xl-ngpus8-seed8620-20221114-191249-EVPtXbhn5qpcyN3Mwfhjio_ckpt_best.pth
CUDA_VISIBLE_DEVICES=1  python examples/segmentation/main.py --cfg cfgs/scannet/pointmetabase-xxl.yaml mode=test dataset.test.split=val --pretrained_path log/scannet/scannet-train-pointmetabase-xxl-ngpus8-seed9599-20221115-100803-YEVwdrsuLRxTcyC2fjoQTZ/checkpoint/scannet-train-pointmetabase-xxl-ngpus8-seed9599-20221115-100803-YEVwdrsuLRxTcyC2fjoQTZ_ckpt_best.pth

# Profile Parameters, FLOPs, and Throughput
CUDA_VISIBLE_DEVICES=3 python examples/profile.py --cfg cfgs/scannet/pointnext-xl.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=3 python examples/profile.py --cfg cfgs/scannet/pointnext-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=3 python examples/profile.py --cfg cfgs/scannet/pointmetabase-l.yaml batch_size=16 num_points=15000 flops=True timing=True
CUDA_VISIBLE_DEVICES=3 python examples/profile.py --cfg cfgs/scannet/pointmetabase-b.yaml batch_size=16 num_points=15000 flops=True timing=True