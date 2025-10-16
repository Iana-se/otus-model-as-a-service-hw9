### ДЗ №9
Действия:
создаю kubernetis кластер.  
Добавила код с моделью регресси, обучением ее на медицинских данных, а после разворачиванием в сервисе в кубере.     
make build-prod.   
make push-prod.  
yc managed-kubernetes cluster get-credentials --id <CUBER_CLUSTER_ID> --external.  
make helm-install-ingress.  
kubectl apply -f k8s/namespace.yaml.        
kubectl config set-context --current --namespace=otus-maas.   
kubectl apply -f k8s/deployment.yaml.    
kubectl apply -f k8s/service.yaml.           
kubectl apply -f k8s/ingress.yaml.    

