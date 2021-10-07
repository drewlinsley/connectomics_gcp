TPU=tf-vm
ZONE=us-central1-a  # europe-west4-a

gcloud alpha compute tpus tpu-vm create $TPU \
--zone=$ZONE \
--accelerator-type=v3-8 \
--version=v2-alpha

gcloud alpha compute tpus tpu-vm ssh $TPU \
  --zone $ZONE

