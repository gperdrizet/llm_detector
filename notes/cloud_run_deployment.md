# Cloud run deployment notes

## TODO

1. Set up artifact registry for container images

After the consultation call with the Google Cloud Customer Engineer and my Startup Success Manager, it seems that the ea the easiest/fastest way to get Agatha into the cloud and running is a 'lift-and-shift' of our Docker containers to Google Cloud Run. We will revisit later to decide if we want to switch to GKE for the re-engineer of the classifier.

So, the containers we need to move are:

1. The Redis task queue
2. The telegram bot
3. The text classification API

Let's dig into the [documentation](https://cloud.google.com/run/docs/overview/what-is-cloud-run) on Google Cloud run and see what we are dealing with.

## Cloud run overview

Can use containers, or have GCR build them for you.

Tasks are run as one of two types:

- **Services**: Used to run code that responds to web requests, events, or functions.
- **Jobs**: Used to run code that performs work (a job) and quits when the work is done.

We obviously want to run services for now, but a job sounds perfect for benchmarking or generating new synthetic data to tune the model.

Cloud run 'scales to zero' i.e. if there are no incoming requests, the instances will be removed. I don't think we want this, because to spin up and API instance, we have to download the LLMs - this takes a while. Look out for a way to keep the API container alive, even if it is not being used.

Payment can be request-based or instance-based. Request-based doesn't charge when a CPU is not allocated, but there is an additional per-request fee. Instance-based just keeps the CPU allocated and charges for the lifetime of the instance. Sounds like we want instance-based due to the above consideration about container initialization time. Unless, the container stays alive when it does not have a CPU allocated.

## [Quickstart: Deploy to Cloud Run](Quickstart: Deploy to Cloud Run)

### 1. Setup

- Sign into Google Cloud
- Create a project: ask-agatha
- Make sure billing is enabled
- Make sure you have the following IAM: `roles/run.developer`, `roles/iam.serviceAccountUser`
- Enable the cloud run API in the project

Can deploy from DockerHub, but Google recommends their own artifact registry. See [Create remote repositories](https://cloud.google.com/artifact-registry/docs/repositories/remote-repo). We will come back and set that up later.

### 2. Deploy container image

#### 2.1. Set-up local shell with gcloud cli

##### 2.1.1. Install gcloud

Documentation [here](https://cloud.google.com/sdk/docs/install). Following the Debian/Ubuntu instructions on *pyrite*:

Update the system:

```bash
sudo apt-get update
sudo apt-get upgrade
```

Make sure that `apt-transport-https` and `curl` are installed:

```bash
sudo apt-get install apt-transport-https ca-certificates gnupg curl
```

Import the Google Cloud public key:

```bash
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
```

Add gcloud as a package source:

```bash
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
```

Update & install:

```bash
sudo apt-get update && sudo apt-get install google-cloud-cli
```

Check the installation:

```bash
$ gcloud --version

Google Cloud SDK 501.0.0
alpha 2024.11.08
beta 2024.11.08
bq 2.1.9
bundled-python3-unix 3.11.9
core 2024.11.08
gcloud-crc32c 1.0.0
gsutil 5.31
```

Initialize the installation:

```bash
gcloud init
```

This brings you to a web login page for a verification code and then ask which GC project you want to associate to. Used *ask-agatha*.

##### 2.1.2. Initialize gcloud

Documentation [here](https://cloud.google.com/sdk/docs/initializing). After running `glcoud init` and following initial set-up instructions, do the following:

Set the default region with:

```bash
gcloud config set run/region us-central1
```

Not much to do - `gcloud init` handled everything. Check the current configuration with:

```bash
$ gcloud config list

[core]
account = gperdrizet@ask-agatha.com
disable_usage_reporting = True
project = ask-agatha

Your active configuration is: [default]
```

OK, looks like we are ready to go with the default configuration.

#### 2.2. Add secrets

The containers need to have access to a few secrets. We are providing these locally from Docker Compose via environment variables:

1. REDIS_PASSWORD
2. HF_TOKEN
3. TELEGRAM_TOKEN

Sounds like the way to do this for Cloud Run is via the Secrets Manager.

- Enable the Secret Manager API via GC console

Create the secrets. Documentation [here](https://cloud.google.com/secret-manager/docs/creating-and-accessing-secrets#secretmanager-create-secret-console).

```bash
printf $REDIS_PASSWORD | gcloud secrets create redis_password --data-file=- --replication-policy="automatic"
printf $HF_TOKEN | gcloud secrets create hf_token --data-file=- --replication-policy="automatic"
printf $TELEGRAM_TOKEN | gcloud secrets create telegram_token --data-file=- --replication-policy="automatic"
```

Create a user managed service account:

```bash
gcloud iam service-accounts create \
  ask-agatha-service \
  --description="Service account for Ask Agatha Cloud Run deployment" \
  --display-name="agatha-service"
```

Then, give the revision service account secretmanager.secretAccessor role.

```bash
gcloud projects add-iam-policy-binding ask-agatha \
  --member="serviceAccount:ask-agatha-service@ask-agatha.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

And set it as the service account for ask-agatha:

```bash
gcloud run services update ask-agatha --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com
```

#### 2.2. Deploy the image(s)

From local shell on *pyrite*:

##### 2.2.1. Telegram bot

```bash
$ gcloud run deploy bot \
  --image gperdrizet/agatha:bot \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-secrets=TELEGRAM_TOKEN=telegram_token:latest \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com

X Deploying...                                                                              
  - Creating Revision...                                                                    
  . Routing traffic...                                                                      
Deployment failed                                                                           
ERROR: (gcloud.run.services.update) Revision 'bot-00005-52q' is not ready and cannot serve traffic. The user-provided container failed to start and listen on the port defined provided by the PORT=8080 environment variable within the allocated timeout. This can happen when the container port is misconfigured or if the timeout is too short. The health check timeout can be extended. Logs for this revision might contain more information.

Logs URL: https://console.cloud.google.com/logs/viewer?project=ask-agatha&resource=cloud_run_revision/service_name/bot/revision_name/bot-00005-52q&advancedFilter=resource.type%3D%22cloud_run_revision%22%0Aresource.labels.service_name%3D%22bot%22%0Aresource.labels.revision_name%3D%22bot-00005-52q%22 
For more troubleshooting guidance, see https://cloud.google.com/run/docs/troubleshooting#container-failed-to-start
```

OK, looking like maybe Cloud Run is not going to be it for us. See the [Container runtime contract](https://cloud.google.com/run/docs/container-contract). A few points to note:

1. Services must listen for requests on a specific port (default 8080) - so, basically you have to run a webserver inside the container which will only be active when there are incoming requests. See note at the beginning of this document about initializing containers.
2. Job execution must exit on completion with exit code 0. This is not great for us either, we could run the API container as a job, but that's kinda hacky and unintended - it's really a service.

Starting to sound more and more like Cloud Run is not for us.

Let's see if we can deploy the API container at all by setting the listen port to the same as the Flask server.

```bash
$ gcloud run deploy bot \
  --image gperdrizet/agatha:bot \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-secrets=TELEGRAM_TOKEN=telegram_token:latest \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --port $FLASK_PORT
  --command "start_api.sh"

"Deploying container to Cloud Run service [bot] in project [ask-agatha] region [us-central1]
  Deploying...                                                                              
X Deploying...                                                                              
  - Creating Revision...                                                                    

Deployment failed                                                                           
ERROR: (gcloud.run.deploy) Revision 'bot-00006-hv8' is not ready and cannot serve traffic. The user-provided container failed to start and listen on the port defined provided by the PORT=5000 environment variable within the allocated timeout. This can happen when the container port is misconfigured or if the timeout is too short. The health check timeout can be extended. Logs for this revision might contain more information.

Logs URL: https://console.cloud.google.com/logs/viewer?project=ask-agatha&resource=cloud_run_revision/service_name/bot/revision_name/bot-00006-hv8&advancedFilter=resource.type%3D%22cloud_run_revision%22%0Aresource.labels.service_name%3D%22bot%22%0Aresource.labels.revision_name%3D%22bot-00006-hv8%22 
For more troubleshooting guidance, see https://cloud.google.com/run/docs/troubleshooting#container-failed-to-start
```

Yep - should have thought of that. To spin up, the container needs to download the models, which takes a few minutes. Could be 