VMNAME=pytorch
TPUNAME=pytorch-tpu
ZONE=us-central1-f  # europe-west4-a
# ZONE=europe-west4-a
# ZONE=us-east1-d
TPU=v3-8


gcloud compute instances delete $TPUNAME --zone=$ZONE --quiet

gcloud alpha compute tpus tpu-vm create $TPUNAME \
--zone=$ZONE \
--accelerator-type=v3-8 \
--version=v2-alpha \
# --boot-disk-size=200GB \

gcloud alpha compute tpus tpu-vm ssh $TPUNAME \
  --zone $ZONE

