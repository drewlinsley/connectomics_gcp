VMNAME=pytorch
TPUNAME=pytorch-tpu
ZONE=us-central1-a  # europe-west4-a
ZONE=europe-west4-a
TPU=v3-8


gcloud compute instances delete $VMNAME --zone=$ZONE --quiet

gcloud compute instances create $VMNAME \
--zone=$ZONE  \
--machine-type=n1-standard-16  \
--image-family=torch-xla \
--image-project=ml-images  \
--boot-disk-size=200GB \
--scopes=https://www.googleapis.com/auth/cloud-platform

gcloud compute tpus create $TPUNAME \
--zone=$ZONE \
--network=default \
--version=pytorch-1.9 \
--accelerator-type=$TPU

gcloud compute ssh $TPUNAME --zone=$ZONE

