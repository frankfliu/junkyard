# gcloud cli

## Initialize gcloud cli

```bash
gcloud init
gcloud config list --format=json
gcloud config set project $PROJECT
gcloud auth application-default set-quota-project $PROJECT
gcloud config set billing/quota_project $PROJECT

gcloud auth application-default login

gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-c
gcloud config set ai/region us-central1

gcloud projects describe $PROJECT --format="value(projectNumber)
````

## metadata server

```bash
curl http://metadata.google.internal/computeMetadata/v1
curl http://169.254.169.254/computeMetadata/v1
```

## GCE

```bash
gcloud compute machine-types list
gcloud compute instances list
gcloud compute images list \
    --project deeplearning-platform-release \
    --format="value(NAME)" \
    --no-standard-images

gcloud compute instances describe XXXX
gcloud compute instances describe XXXX --format="yaml(serviceAccounts)"
gcloud compute instances set-service-account XXXX \
    --service-account=1095264877375-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_write,default

gcloud compute instances start XXXX

gcloud compute firewall-rules list | grep "tcp:22"
```

## GCS

```bash
gcloud storage buckets list
gcloud storage ls gs://$MY_BUCKET

gcloud storage rsync --recursive --no-ignore-symlinks . gs://$MY_BUCKET/target/

gcloud storage ls gs://$MY_BUCKET --soft-deleted --recursive
gcloud storage restore gs://$MY_BUCKET/deleted.txt#1745437592030740

gcsfuse --implicit-dirs $MY_BUCKET $HOME/mount/folder
gcsfuse --foreground --log-severity=TRACE --implicit-dirs $MY_BUCKET $HOME/mount/folder
fusermount -u $HOME/mount/folder
```

## environment variables

export CLOUDSDK_API_ENDPOINT_OVERRIDES_AIPLATFORM=https://us-central1-staging-aiplatform.sandbox.googleapis.com/
export CLOUDSDK_API_ENDPOINT_OVERRIDES_AIPLATFORM=https://us-central1-autopush-aiplatform.sandbox.googleapis.com/

## IAM

```bash
gcloud iam service-accounts list
gcloud iam service-accounts get-iam-policy $SERVICE_ACCOUNT
gcloud projects get-iam-policy $PROJECT
```

## GAR

```bash
gcloud auth configure-docker us-docker.pkg.dev
cat ~/.docker/config.json

gcloud artifacts repositories list --project=$PROJECT --location=us

gcloud artifacts repositories create $MY_REPO --repository-format=Docker --location=us
gcloud artifacts repositories describe $MY_REPO --location=us

gcloud artifacts docker images list us-docker.pkg.dev/$PROJECT/$MY_REPO
gcloud artifacts docker tag list us-docker.pkg.dev/$PROJECT/$MY_REPO

gcloud artifacts repositories add-iam-policy-binding $MY_REPO \
   --location=us --member=domain:google.com --role=roles/artifactregistry.reader

# clean up policy
gcloud artifacts repositories list-cleanup-policies $MY_REPO --location=us
gcloud artifacts repositories set-cleanup-policies $MY_REPO \
   --project=$PROJECT \
   --location=us \
   --policy=$HOME/policy.json \
   --dry-run
```

## cloud run

```bash
gcloud run services list

time gcloud run deploy $MY_CLOUD_RUN --region=$LOCATION \
    --no-allow-unauthenticated \
    --container=primary \
    --image us-docker.pkg.dev/$PROJECT/... \
    --port=8080 \
    --args=serve \
    --set-env-vars="env1=..." \
    --startup-probe="httpGet.port=8080,httpGet.path=/ping" \
    --cpu=8 \
    --memory=8Gi
```

# logging

```bash
gcloud logging logs list
gcloud logging logs delete projects/$PROJECT/logs/cloudbuild
gcloud alpha logging tail "severity>=ERROR"

gcloud logging read 'protoPayload.serviceName="artifactregistry.googleapis.com" AND protoPayload.request.parent="
projects/$PROJECT_ID/locations/$LOCATION/repositories/$REPOSITORY" AND protoPayload.request.validateOnly=true' \
    --resource-names="projects/$PROJECT" \
    --project=$PROJECT

gcloud alpha logging read logName=projects/$PROJECT/logs/aiplatform.googleapis.com%2Fprediction_container
```

## GKE

```bash
gcloud components install gke-gcloud-auth-plugin
gcloud components install kubectl

CLUSTER_NAME=$(gcloud container clusters list --project $TP --format="[no-heading](NAME)")
echo $CLUSTER_NAME

gcloud container clusters get-credentials $CLUSTER_NAME --region us-central1 --project $TP

kubectl get namespaces

POD=$(kubectl get pods --all-namespaces | grep prediction | awk '{print $2}')
echo $POD

kubectl exec -it $POD -n prediction -- /bin/bash

gcloud beta container clusters delete cluster-1 --zone us-central1-b

gcloud beta container clusters create cluster-1 \
    --project $PROJECT \
    --zone us-central1-b \
    --fleet-project=$PROJECT\
    --tier "standard" \
    --no-enable-basic-auth \
    --cluster-version "1.30.5-gke.1443001" \
    --release-channel "regular" \
    --machine-type "g2-standard-4" \
    --accelerator "type=nvidia-l4,count=1,gpu-driver-version=latest" \
    --image-type "COS_CONTAINERD" \
    --disk-type "pd-ssd" --disk-size "300" \
    --metadata disable-legacy-endpoints=true \
    --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
    --num-nodes "2" \
    --logging=SYSTEM,WORKLOAD \
    --monitoring=SYSTEM,STORAGE,POD,DEPLOYMENT,STATEFULSET,DAEMONSET,HPA,CADVISOR,KUBELET \
    --enable-ip-alias \
    --network "projects/$PROJECT/global/networks/default" \
    --subnetwork "projects/$PROJECT/regions/us-central1/subnetworks/default" \
    --no-enable-intra-node-visibility \
    --default-max-pods-per-node "110" \
    --enable-ip-access \
    --security-posture=standard \
    --workload-vulnerability-scanning=disabled \
    --no-enable-master-authorized-networks \
    --no-enable-google-cloud-access \
    --addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver \
    --enable-autoupgrade --enable-autorepair \
    --max-surge-upgrade 1 \
    --max-unavailable-upgrade 0 \
    --binauthz-evaluation-mode=DISABLED \
    --enable-managed-prometheus \
    --enable-shielded-nodes \
    --no-autoprovisioning-enable-insecure-kubelet-readonly-port

gcloud container clusters get-credentials cluster-1 \
  --region us-central1

kubectl get deployment deployment-1 -o yaml

kubectl get pods --namespace $NAMESPACE
kubectl wait --for=condition=Available --timeout=700s --namespace $NAMESPACE deployment/tei-deployment

kubectl port-forward --namespace $NAMESPACE service/tei-service 8080:8080

gcloud container get-server-config \
    --flatten="channels" \
    --filter="channels.channel=STABLE" \
    --format="yaml(channels.channel,channels.defaultVersion,channels.validVersions)"
```

# Pub/Sub

```bash
gcloud pubsub topics create gmail_notification
gcloud pubsub subscriptions create gmail_notification-sub --topic gmail_notification

gcloud pubsub subscriptions delete gmail_notification-sub
gcloud pubsub topics delete gmail_notification

gcloud pubsub topics publish gmail_notification --message="hello"
gcloud pubsub subscriptions pull gmail_notification-sub --auto-ack
```
