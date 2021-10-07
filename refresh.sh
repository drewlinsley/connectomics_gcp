git commit -am "Updated."
git push
kubectl delete job celltype-train-tpu
bash train_celltype_model.sh
kubectl get pods -w

