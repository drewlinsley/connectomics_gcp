## Run jobs on the GCP cluster
# Add your google credentials to your .bash_profile
`https://cloud.google.com/docs/authentication/getting-started`

# Install TPU unicorn for babysitting TPUs
`https://github.com/shawwn/tpunicorn`

# Create a cluster, generate and run experiments, then delete cluster
bash create_cluster.sh
python prepare_experiments.py  --exp=experiments/bu_td_attractive_repulsive.yaml
bash run_kube_exps.sh

# Check kube status
kubectl get pods -w

# Check pod logs
kubectl logs <pod-name>
kubectl logs --follow <pod-name>

# Babysit existing preemptibles
bash babysit_tpus.sh

# Run tensorboard on the cluster
kubectl run tensorboard \
  --image tensorflow/tensorflow:2.2.0 \
  --port 6006 \
  -- bash -c "pip install tensorboard-plugin-profile==2.2.0 cloud-tpu-client && curl -i icanhazip.com && tensorboard --logdir=gs://serrelab/prj-selfsup --bind_all"
kubectl get pod tensorboard -w
kubectl port-forward pod/tensorboard 6006  # Access the TB at http://localhost:6006
kubectl delete pod tensorboard

# Delete pods in the cluster
kubectl delete pods <pod-name>
kubectl delete --all pods
kubectl get job --all-namespaces
kubectl delete job hmax-tpu-v3-256

# Clean up cluster
bash stop_babysitting.sh
bash delete_cluster.sh

# Monitor your kube
`https://console.cloud.google.com/monitoring`

# Run a single kube job
kubectl create -f kube_job.yaml

## Run individual jobs
# Train a model on ILSVRC12 on the vm
bash jobs/pretrain_ilsrc.sh 16 ar ar prj-selfsup-v2-22

# Create a tensorboard
tensorboard --logdir=$(cat current_job.txt) &
bash get_ip.sh  # navigate to <ip>:6006 in your web browser
