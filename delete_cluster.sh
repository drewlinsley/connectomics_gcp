

CLUSTER_NAME="`grep CLUSTER_NAME create_cluster.sh | head -1`"
CLUSTER_NAME=(${CLUSTER_NAME//=/ })
CLUSTER_NAME=${CLUSTER_NAME[1]}
echo "Deleting cluster ${CLUSTER_NAME}" 

gcloud container clusters delete $CLUSTER_NAME --project=beyond-dl-1503610372419 --zone=europe-west4-a
