# Cloud run deployment notes

## TODO

1. Set up artifact registry for container images

After the consultation call with the Google Cloud Customer Engineer and my Startup Success Manager, it seems that the easiest/fastest way to get Agatha into the cloud and running is a 'lift-and-shift' of our Docker containers to Google Cloud Run. We will revisit later to decide if we want to switch to GKE for the re-engineer of the classifier.

So, the containers we need to move are:

1. The Redis task queue
2. The Telegram bot
3. The text classification API

Let's dig into the [documentation](https://cloud.google.com/run/docs/overview/what-is-cloud-run) on Google Cloud run and see what we are dealing with.

## Cloud run overview

Can use pre-build Docker containers, or have GCR build them for you.

Tasks are run as one of two types:

- **Services**: Used to run code that responds to web requests, events, or functions.
- **Jobs**: Used to run code that performs work (a job) and quits when the work is done.

We obviously want to run services for now, but a job sounds perfect for benchmarking or generating new synthetic data to tune the model.

Cloud run 'scales to zero' i.e. if there are no incoming requests, the instances will be removed. I don't think we want this, because to spin up and API instance, we have to download the LLMs - this takes a while. Look out for a way to keep the API container alive, even if it is not being used.

Payment can be request-based or instance-based. Request-based doesn't charge when a CPU is not allocated, but there is an additional per-request fee. Instance-based just keeps the CPU allocated and charges for the lifetime of the instance. Sounds like we want instance-based due to the above consideration about container initialization time. Unless, the container stays alive when it does not have a CPU allocated.

## [Quickstart: Deploy to Cloud Run](https://cloud.google.com/run/docs/quickstarts/deploy-container)

## 1. Setup

- Sign in to Google Cloud
- Create a project: ask-agatha
- Make sure billing is enabled
- Make sure you have the following IAM: `roles/run.developer`, `roles/iam.serviceAccountUser`
- Enable the cloud run API in the project

Can deploy from DockerHub, but Google recommends their own artifact registry. See [Create remote repositories](https://cloud.google.com/artifact-registry/docs/repositories/remote-repo). We will come back and set that up later.

## 2. Deploy container image

### 2.1. Set-up local shell with gcloud cli

#### 2.1.1. Install gcloud

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

#### 2.1.2. Initialize gcloud

Documentation [here](https://cloud.google.com/sdk/docs/initializing). After running `gcloud init` and following initial set-up instructions, do the following:

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

### 2.2. Add secrets

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

### 2.2. Deploy the image(s)

From local shell on *pyrite*:

#### 2.2.1. Text classification API

```bash
$ gcloud run deploy api \
  --image gperdrizet/agatha:api \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-secrets=HF_TOKEN=hf_token:latest \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --command "./start_api.sh"

Deploying container to Cloud Run service [api] in project [ask-agatha] region [us-central1]
X Deploying...                                                                                                              
  - Creating Revision...                                                                                                    
  . Routing traffic...                                                                                                      
Deployment failed                                                                                                           
ERROR: (gcloud.run.deploy) Revision 'api-00002-sh2' is not ready and cannot serve traffic. The user-provided container failed to start and listen on the port defined provided by the PORT=8080 environment variable within the allocated timeout. This can happen when the container port is misconfigured or if the timeout is too short. The health check timeout can be extended. Logs for this revision might contain more information.

Logs URL: https://console.cloud.google.com/logs/viewer?project=ask-agatha&resource=cloud_run_revision/service_name/api/revision_name/api-00002-sh2&advancedFilter=resource.type%3D%22cloud_run_revision%22%0Aresource.labels.service_name%3D%22api%22%0Aresource.labels.revision_name%3D%22api-00002-sh2%22 
For more troubleshooting guidance, see https://cloud.google.com/run/docs/troubleshooting#container-failed-to-start
```

OK, looking like maybe Cloud Run is not going to be it for us. See the [Container runtime contract](https://cloud.google.com/run/docs/container-contract). A few points to note:

1. Services must listen for requests on a specific port (default 8080) - so, basically you have to run a web server inside the container which will only be active when there are incoming requests. See note at the beginning of this document about initializing containers.
2. Job execution must exit on completion with exit code 0. This is not great for us either, we could run the API container as a job, but that's kinda hacky and unintended - it's really a service.

Starting to sound more and more like Cloud Run is not for us.

Let's see if we can deploy the API container at all by setting the listen port to the same as the Flask server.

```bash
$ gcloud run deploy api \
  --image gperdrizet/agatha:api \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-secrets=HF_TOKEN=hf_token:latest \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --command "./start_api.sh" \
  --port $FLASK_PORT

Deploying container to Cloud Run service [api] in project [ask-agatha] region [us-central1]
X Deploying...                                                                                                              
  - Creating Revision...                                                                                                    
  . Routing traffic...                                                                                                      
Deployment failed                                                                                                           
ERROR: (gcloud.run.deploy) Revision 'api-00003-sxg' is not ready and cannot serve traffic. The user-provided container failed to start and listen on the port defined provided by the PORT=5000 environment variable within the allocated timeout. This can happen when the container port is misconfigured or if the timeout is too short. The health check timeout can be extended. Logs for this revision might contain more information.

Logs URL: https://console.cloud.google.com/logs/viewer?project=ask-agatha&resource=cloud_run_revision/service_name/api/revision_name/api-00003-sxg&advancedFilter=resource.type%3D%22cloud_run_revision%22%0Aresource.labels.service_name%3D%22api%22%0Aresource.labels.revision_name%3D%22api-00003-sxg%22 
For more troubleshooting guidance, see https://cloud.google.com/run/docs/troubleshooting#container-failed-to-start
```

#### 2.2.2. Troubleshooting

OK, so could be any number of problems:

1. The API container image is base on nvidia/cuda:11.4.3-runtime-ubuntu20.04, so it could be the fact that we don't have a GPU.
2. On first spin-up we download the models, which could be exceeding the startup timeout.
3. There could be other problems inside the container, for example, the redis server is not running....

Let's see if we can get anything useful from the logs - nope not, really. Would love to see the logs from inside the container. Not sure how to do that. Let's start doing things we know we need...

#### 2.2.3. Attach GPU(s)

Following the [GPU (services) documentation](https://cloud.google.com/run/docs/configuring/services/gpu).

##### 2.2.3.1. Configure CPU always allocated

```bash
gcloud run services update api --no-cpu-throttling
```

**Note**: can also be set during deployment with by passing `--no-cpu-throttling`.

##### 2.2.3.2. Configure 8 CPUs

```bash
gcloud run services update api --cpu 8
```

**Note**: can also be set during deployment with by passing `--cpu`.

##### 2.2.3.3. Configure 32 GB memory

```bash
gcloud run services update api --memory 32G
```

**Note**: can also be set during deployment with by passing `--memory`.

##### 2.2.3.4. Configure concurrency

Let's set for 2, that's the most workers we can handle on a single GPU.

```bash
gcloud run services update api --concurrency 2
```

**Note**: can also be set during deployment with by passing `--concurrency`.

##### 2.2.3.5. Configure max instances

```bash
gcloud run services update api --max-instances 1
```

**Note**: can also be set during deployment with by passing `--max-instances`.

##### 2.2.3.6. Deploy with GPU

Put it all together:

```bash
$ gcloud beta run deploy api \
  --image gperdrizet/agatha:api \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-secrets=HF_TOKEN=hf_token:latest \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --command "./start_api.sh" \
  --port $FLASK_PORT \
  --no-cpu-throttling \
  --cpu 8 \
  --memory 32G \
  --concurrency 2 \
  --max-instances 1 \
  --gpu 1 \
  --gpu-type "nvidia-l4"

  Deploying container to Cloud Run service [api] in project [ask-agatha] region [us-central1]
X Deploying...                                                                                                                                      
  - Creating Revision...                                                                                                                            
  . Routing traffic...                                                                                                                              
Deployment failed                                                                                                                                   
ERROR: (gcloud.beta.run.deploy) Revision 'api-00006-jjh' is not ready and cannot serve traffic. The user-provided container failed to start and listen on the port defined provided by the PORT=5000 environment variable within the allocated timeout. This can happen when the container port is misconfigured or if the timeout is too short. The health check timeout can be extended. Logs for this revision might contain more information.

Logs URL: https://console.cloud.google.com/logs/viewer?project=ask-agatha&resource=cloud_run_revision/service_name/api/revision_name/api-00006-jjh&advancedFilter=resource.type%3D%22cloud_run_revision%22%0Aresource.labels.service_name%3D%22api%22%0Aresource.labels.revision_name%3D%22api-00006-jjh%22 
For more troubleshooting guidance, see https://cloud.google.com/run/docs/troubleshooting#container-failed-to-start
```

OK, logs not helpful. I'm a little worried that it's not starting because redis is not there. Let's try just starting the redis container. After that, I think we need some networking setup to get inter-container communication working.

#### 2.2.4. Redis container

```bash
$ gcloud beta run deploy redis \
  --image gperdrizet/agatha:redis \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --command "./start_server.sh" \
  --port $REDIS_PORT

Deploying container to Cloud Run service [redis] in project [ask-agatha] region [us-central1]
X Deploying new service...                                                                                                                          
  - Creating Revision...                                                                                                                            
  . Routing traffic...                                                                                                                              
Deployment failed                                                                                                                                   
ERROR: (gcloud.beta.run.deploy) Revision 'redis-00001-qz5' is not ready and cannot serve traffic. The user-provided container failed to start and listen on the port defined provided by the PORT=6379 environment variable within the allocated timeout. This can happen when the container port is misconfigured or if the timeout is too short. The health check timeout can be extended. Logs for this revision might contain more information.

Logs URL: https://console.cloud.google.com/logs/viewer?project=ask-agatha&resource=cloud_run_revision/service_name/redis/revision_name/redis-00001-qz5&advancedFilter=resource.type%3D%22cloud_run_revision%22%0Aresource.labels.service_name%3D%22redis%22%0Aresource.labels.revision_name%3D%22redis-00001-qz5%22 
For more troubleshooting guidance, see https://cloud.google.com/run/docs/troubleshooting#container-failed-to-start
```

Yeah, nope - not sure how to troubleshoot this - maybe redis is not responding to the health check? There has gotta be a way to see logs from inside the container...

OK, here we go:

```bash
$ gcloud run services logs read redis --limit=10 --project ask-agatha

024-11-17 04:41:23 sysctl: write error: I/O error
2024-11-17 04:41:23 3:C 17 Nov 2024 04:41:23.477 # WARNING Memory overcommit must be enabled! Without it, a background save or replication may fail under low memory condition. Being disabled, it can also cause failures without low memory condition, see https://github.com/jemalloc/jemalloc/issues/1328. To fix this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory=1' for this to take effect.
2024-11-17 04:41:23 3:M 17 Nov 2024 04:41:23.480 # WARNING: The TCP backlog setting of 511 cannot be enforced because /proc/sys/net/core/somaxconn is set to the lower value of 128.
2024-11-17 04:41:28 3:M 17 Nov 2024 04:41:28.483 # Warning: Could not create server TCP listening socket -requirepass:6379: Try again
2024-11-17 04:41:28 3:M 17 Nov 2024 04:41:28.483 # Failed listening on port 6379 (tcp), aborting.
```

Thought I fixed the vm.overcommit warning - maybe there are new permissions issues on Cloud Run. I'm more interested in the line: `Warning: Could not create server TCP listening socket -requirepass:6379: Try again`. Hmm, let's try a different port? How about the default 8080 for https:

```bash
$ export REDIS_PORT='8080'
$ echo $REDIS_PORT
8080
```

Ok, try again...

```bash
$ gcloud beta run deploy redis \
  --image gperdrizet/agatha:redis \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --command "./start_server.sh" \
  --port $REDIS_PORT

  Deploying container to Cloud Run service [redis] in project [ask-agatha] region [us-central1]
X Deploying...                                                                                                                                      
  - Creating Revision...                                                                                                                            
  . Routing traffic...                                                                                                                              
Deployment failed                                                                                                                                   
ERROR: (gcloud.beta.run.deploy) Revision 'redis-00002-9j7' is not ready and cannot serve traffic. The user-provided container failed to start and listen on the port defined provided by the PORT=8080 environment variable within the allocated timeout. This can happen when the container port is misconfigured or if the timeout is too short. The health check timeout can be extended. Logs for this revision might contain more information.

Logs URL: https://console.cloud.google.com/logs/viewer?project=ask-agatha&resource=cloud_run_revision/service_name/redis/revision_name/redis-00002-9j7&advancedFilter=resource.type%3D%22cloud_run_revision%22%0Aresource.labels.service_name%3D%22redis%22%0Aresource.labels.revision_name%3D%22redis-00002-9j7%22 
For more troubleshooting guidance, see https://cloud.google.com/run/docs/troubleshooting#container-failed-to-start
```

How about...

```bash
$ gcloud run services logs read redis --limit=100 --project ask-agatha

2024-11-17 04:41:23 sysctl: write error: I/O error
2024-11-17 04:41:23 3:C 17 Nov 2024 04:41:23.477 # WARNING Memory overcommit must be enabled! Without it, a background save or replication may fail under low memory condition. Being disabled, it can also cause failures without low memory condition, see https://github.com/jemalloc/jemalloc/issues/1328. To fix this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory=1' for this to take effect.
2024-11-17 04:41:23 3:M 17 Nov 2024 04:41:23.480 # WARNING: The TCP backlog setting of 511 cannot be enforced because /proc/sys/net/core/somaxconn is set to the lower value of 128.
2024-11-17 04:41:28 3:M 17 Nov 2024 04:41:28.483 # Warning: Could not create server TCP listening socket -requirepass:6379: Try again
2024-11-17 04:41:28 3:M 17 Nov 2024 04:41:28.483 # Failed listening on port 6379 (tcp), aborting.
2024-11-17 04:54:45 sysctl: write error: I/O error
2024-11-17 04:54:45 3:C 17 Nov 2024 04:54:45.617 # WARNING Memory overcommit must be enabled! Without it, a background save or replication may fail under low memory condition. Being disabled, it can also cause failures without low memory condition, see https://github.com/jemalloc/jemalloc/issues/1328. To fix this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory=1' for this to take effect.
2024-11-17 04:54:45 3:M 17 Nov 2024 04:54:45.620 # WARNING: The TCP backlog setting of 511 cannot be enforced because /proc/sys/net/core/somaxconn is set to the lower value of 128.
2024-11-17 04:54:50 3:M 17 Nov 2024 04:54:50.623 # Warning: Could not create server TCP listening socket -requirepass:6379: Try again
2024-11-17 04:54:50 3:M 17 Nov 2024 04:54:50.623 # Failed listening on port 6379 (tcp), aborting.
```

Nope. What's the deal with `WARNING: The TCP backlog setting of 511 cannot be enforced because /proc/sys/net/core/somaxconn is set to the lower value of 128.`?

OK, looking at stuff - I realize that we set a static redis ip via docker compose - so that is not available to Cloud Run. Maybe the permisssions problem is redis trying to use a disallowed ip in the Cloud Run container. Let's try setting it:

```bash
$ export REDIS_IP='0.0.0.0'
$ echo $REDIS_IP
8080
```

Now try passing it:

```bash
$ gcloud beta run deploy redis \
  --image gperdrizet/agatha:redis \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-env-vars REDIS_PORT=$REDIS_PORT \
  --port $REDIS_PORT \
  --command "./start_server.sh" \
  --no-cpu-throttling
```

Still no - same problems. Starting to think this is just messy/bad configuration and we should fix it at the image build level. The fact that the vm.overcommit warning is back means that things we think we are doing are not working. See the container logs:

```text
2024-11-18 14:31:58 vm.overcommit_memory = 1
2024-11-18 14:32:03 14:M 18 Nov 2024 14:32:03.967 # Warning: Could not create server TCP listening socket -requirepass:6379: Try again
2024-11-18 14:32:03 14:M 18 Nov 2024 14:32:03.967 # Failed listening on port 6379 (tcp), aborting.
```

Ok, looks like that handled the vm.overcommit_memory issue. But, I don't understand why we can't listen on 6379 - I'm pretty sure that there is no ufw or anything inside the container. OK, have an idea, look at this:

```bash
$ echo $REDIS_IP
192.168.1.148
```

We are setting the wrong listen IP via the host environment variable on **pyrite**. Updated `.venv/bin/activate` with `0.0.0.0`. We may also need to update the image - we are using container names via Docker Compose for networking between containers. I think the right way to do this for Cloud Run is make the API the service and run Redis and the bot as 'sidecars'. Let's try with the updated IP and see how the logs change. Hopefully we can get Redis started. Then we can think about the other containers - I don't think the API will start correctly if it can't reach redis.

```bash
$ ./cloud_run_deploy.sh

Deploying container to Cloud Run service [redis] in project [ask-agatha] region [us-central1]
X Deploying...                                                                                               
  - Creating Revision...                                                                                     
  . Routing traffic...                                                                                       
Deployment failed                                                                                            
ERROR: (gcloud.beta.run.deploy) Revision 'redis-00006-g9c' is not ready and cannot serve traffic. The user-provided container failed to start and listen on the port defined provided by the PORT=6379 environment variable within the allocated timeout. This can happen when the container port is misconfigured or if the timeout is too short. The health check timeout can be extended. Logs for this revision might contain more information.

Logs URL: https://console.cloud.google.com/logs/viewer?project=ask-agatha&resource=cloud_run_revision/service_name/redis/revision_name/redis-00006-g9c&advancedFilter=resource.type%3D%22cloud_run_revision%22%0Aresource.labels.service_name%3D%22redis%22%0Aresource.labels.revision_name%3D%22redis-00006-g9c%22 
For more troubleshooting guidance, see https://cloud.google.com/run/docs/troubleshooting#container-failed-to-start
```

Still no, let's check the container logs:

```bash
2024-11-18 14:46:27 vm.overcommit_memory = 1
2024-11-18 14:46:32 14:M 18 Nov 2024 14:46:32.548 # Warning: Could not create server TCP listening socket -requirepass:6379: Try again
2024-11-18 14:46:32 14:M 18 Nov 2024 14:46:32.548 # Failed listening on port 6379 (tcp), aborting.
```

Still the exact same thing. I wonder if it is actually getting updated - we didn't change the container at all, maybe it's not picking up the change in IP? Try deleting the container via the Cloud Console. More problems. Now the memory overcommit warning is back? Ugh:

```text
2024-11-18 14:53:01 sysctl: write error: I/O error
2024-11-18 14:53:01 3:C 18 Nov 2024 14:53:01.863 # WARNING Memory overcommit must be enabled! Without it, a background save or replication may fail under low memory condition. Being disabled, it can also cause failures without low memory condition, see https://github.com/jemalloc/jemalloc/issues/1328. To fix this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory=1' for this to take effect.
2024-11-18 14:53:01 3:M 18 Nov 2024 14:53:01.866 # WARNING: The TCP backlog setting of 511 cannot be enforced because /proc/sys/net/core/somaxconn is set to the lower value of 128.
2024-11-18 14:53:06 3:M 18 Nov 2024 14:53:06.868 # Warning: Could not create server TCP listening socket -requirepass:6379: Try again
2024-11-18 14:53:06 3:M 18 Nov 2024 14:53:06.868 # Failed listening on port 6379 (tcp), aborting.
```

The inconsistency with vm.overcommit is bothering me, but the real problem is port 6379 - maybe let's try re-building the image? Here are the relevant files:

#### start_server.sh

This is the entrypoint - it is run via `gcloud run deploy` using the `--command` flag.

```bash
#!/bin/sh

# Set memory overcommit
sysctl vm.overcommit_memory=1

# Start redis server
redis-server /usr/local/etc/redis/redis.conf \
--loglevel warning \
--bind $REDIS_IP \
--requirepass $REDIS_PASSWORD
```

And here is the Dockerfile used to build the image:

```Dockerfile
FROM redis:7.2-alpine

# Move our redis.conf in
COPY ./redis.conf /usr/local/etc/redis/redis.conf

# Move the server start helper script in
WORKDIR /redis
COPY ./start_server.sh .
```

So, it looks like we are setting the bind IP via the environment variable like we thought, but we don't need the port number inside the container - we are just using the default. But we do have to tell Cloud Run what port the server is on. Added an echo of REDIS_PORT and REDIS_IP to `start_server.sh` and rebuilt and pushed the container to DockerHub. Try again:

```bash
$ ./cloud_run_deploy.sh

Deploying container to Cloud Run service [redis] in project [ask-agatha] region [us-central1]
X Deploying new service...                                                                                   
  - Creating Revision...                                                                                     
  . Routing traffic...                                                                                       
    Setting IAM Policy...                                                                                    
Deployment failed                                                                                            
  Setting IAM policy failed, try "gcloud beta run services add-iam-policy-binding --region=us-central1 --member=allUsers --role=roles/run.invoker redis"
ERROR: (gcloud.run.deploy) Revision 'redis-00001-9zx' is not ready and cannot serve traffic. The user-provided container failed to start and listen on the port defined provided by the PORT=6379 environment variable within the allocated timeout. This can happen when the container port is misconfigured or if the timeout is too short. The health check timeout can be extended. Logs for this revision might contain more information.

Logs URL: https://console.cloud.google.com/logs/viewer?project=ask-agatha&resource=cloud_run_revision/service_name/redis/revision_name/redis-00001-9zx&advancedFilter=resource.type%3D%22cloud_run_revision%22%0Aresource.labels.service_name%3D%22redis%22%0Aresource.labels.revision_name%3D%22redis-00001-9zx%22 
For more troubleshooting guidance, see https://cloud.google.com/run/docs/troubleshooting#container-failed-to-start

$ gcloud run services logs read redis --limit=100 --project ask-agatha

2024-11-18 15:07:35 sysctl: write error: I/O error
2024-11-18 15:07:35 3:C 18 Nov 2024 15:07:35.569 # WARNING Memory overcommit must be enabled! Without it, a background save or replication may fail under low memory condition. Being disabled, it can also cause failures without low memory condition, see https://github.com/jemalloc/jemalloc/issues/1328. To fix this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory=1' for this to take effect.
2024-11-18 15:07:35 3:M 18 Nov 2024 15:07:35.572 # WARNING: The TCP backlog setting of 511 cannot be enforced because /proc/sys/net/core/somaxconn is set to the lower value of 128.
2024-11-18 15:07:40 3:M 18 Nov 2024 15:07:40.575 # Warning: Could not create server TCP listening socket -requirepass:6379: Try again
2024-11-18 15:07:40 3:M 18 Nov 2024 15:07:40.575 # Failed listening on port 6379 (tcp), aborting.
```

Looks like we failed setting the IAM policy. Try again answering 'no' to `Allow unauthenticated invocations to [redis] (y/N)?`. Ok, that fixed the IAM error, but the container logs are still the same. Maybe it's an issue with exposing that port. When running the containers locally via Docker compose we do:

```text
    ports:
      - '6379:6379'
```

Maybe we need to specify this via the Dockerfile? Let's try adding `EXPOSE 6379` to the Dockerfile and re-building the image.

Nope - no help. Exact same issue. Don't think `EXPOSE` is sufficient. If I were going to run this container locally with docker I would add `-p 6379:6379` to map the port - do I need to do that somehow with gcloud?

## 3. Back-up

OK, this is starting to turn into a mess. Let's back up and double check the containers. Then let's look at the Google Cloud Run 'sidecar' pattern. It appears to use a yaml file much like Docker Compose to deploy multiple containers with one main 'ingress' container and other 'sidecar' containers. This might alleviate some issues with containers not starting because services aren't present. But before we do that, let's rebuild each container image and test them locally.

The following environment variables are set via our virtualenv on the local machine:

```bash
export HF_HOME="/mnt/fast_scratch/huggingface_transformers_cache"
export HF_TOKEN="xxx"
export HOST_IP="0.0.0.0"
export REDIS_IP="0.0.0.0"
export FLASK_PORT="5000"
export REDIS_PORT="6379"
export REDIS_PASSWORD="xxx"
export TELEGRAM_TOKEN="xxx"
```

### 3.1. Redis

Dockerfile:

```Dockerfile
FROM redis:7.2-alpine

# Move our redis.conf in
COPY ./redis.conf /usr/local/etc/redis/redis.conf

# Move the server start helper script in
WORKDIR /redis
COPY ./start_server.sh .

CMD ['./start_server.sh']
```

start_server.sh:

```bash
#!/bin/sh

echo $REDIS_PORT
echo $REDIS_IP

# Set memory overcommit
sysctl vm.overcommit_memory=1

# Start redis server
redis-server /usr/local/etc/redis/redis.conf \
--loglevel warning \
--bind $REDIS_IP \
--requirepass $REDIS_PASSWORD
```

### 3.2. Telegram bot

Dockerfile:

```Dockerfile
FROM python:3.10-slim-bookworm

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Update & install python 3.8 & pip
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python3 python3-pip
RUN python3 -m pip install --upgrade pip

# Set the working directory and move the source code in
WORKDIR /agatha_bot
COPY . /agatha_bot

# Install dependencies
RUN pip install -r requirements.txt

CMD ["python3", "./bot.py"]
```

The start-up command runs the bot directly.

### 3.2. Agatha API

Dockerfile:

```Dockerfile
FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Update & install python 3.8 & pip
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python3 python3-pip
RUN python3 -m pip install --upgrade pip

# Set the woring directory and move the source code in
WORKDIR /agatha_api
COPY . /agatha_api

# Install dependencies
RUN pip install -r requirements.txt

# Install bitsandbytes
WORKDIR /agatha_api/bitsandbytes-0.42.0
RUN python3 setup.py install

# Clean up
RUN rm -r /agatha_api/bitsandbytes-0.42.0

# Set the working directory back
WORKDIR /agatha_api

CMD ["./start_api.sh"]
```

start_api.sh:

```bash
#!/bin/bash

# Authenticate session to HuggingFace so we can download gated models if needed
python3 -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('$HF_TOKEN')"

# Start the API with Gunicorn
gunicorn -w 1 --bind $HOST_IP:$FLASK_PORT 'api:flask_app'
```

### 3.3. Build & push

Build the image and push to Docker Hub:

```bash
build_docker_images.sh
docker push gperdrizet/agatha:redis
docker push gperdrizet/agatha:bot
docker push gperdrizet/agatha:api
```

### 3.4. Docker Compose

Run all three containers via Docker Compose.

compose.yaml:

```yaml
name: agatha

services:

  redis:
    container_name: redis
    image: gperdrizet/agatha:redis
    restart: unless-stopped
    environment:
      REDIS_IP: $REDIS_IP
      REDIS_PORT: $REDIS_PORT
      REDIS_PASSWORD: $REDIS_PASSWORD
    ports:
      - $REDIS_PORT:$REDIS_PORT
    privileged: true

  agatha_api:
    container_name: agatha_api
    image: gperdrizet/agatha:api
    restart: unless-stopped
    environment:
      HOST_IP: $HOST_IP
      FLASK_PORT: $FLASK_PORT
      REDIS_IP: $REDIS_IP
      REDIS_PORT: $REDIS_PORT
      HF_TOKEN: $HF_TOKEN
      REDIS_PASSWORD: $REDIS_PASSWORD
    ports:
      - $FLASK_PORT:$FLASK_PORT
    deploy:
      resources:
        reservations:
          devices:
          - driver: nvidia
            device_ids: ['0', '1', '2']
            capabilities: [gpu]

  agatha_bot:
    container_name:  agatha_bot
    image: gperdrizet/agatha:bot
    restart: unless-stopped
    environment:
      HOST_IP: agatha_api
      FLASK_PORT: $FLASK_PORT
      TELEGRAM_TOKEN: $TELEGRAM_TOKEN
```

Everything looks great, all container start with no errors and Agatha works. The only possible issue I see going into this is that the API container takes a wild to spin up - it needs to download the nltk data and both models. But, I do see that gunicorn is listening on `http://0.0.0.0:5000` before all of that starts. Hopefully the container passes the Cloud Run health check.

### 3.5. Cloud Run with sidecar

OK, now we will try and do the same thing via cloud run with sidecar. Here is the configuration file:

sidecar.yaml

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  annotations:
  name: agatha
  labels:
    cloud.googleapis.com/location: "us-central1"
spec:
  template:
    spec:
      containers:
        - image: "gperdrizet/agatha:api"
          ports:
            - containerPort: 5000
          env:
            - name: HOST_IP
              value: $HOST_IP
            - name: FLASK_PORT
              value: $FLASK_PORT
            - name: REDIS_IP
              value: $REDIS_IP
            - name: REDIS_PORT
              value: $REDIS_PORT
            - name: HF_TOKEN
              value: $HF_TOKEN
            - name: REDIS_PASSWORD
              value: $REDIS_PASSWORD

        - image: "gperdrizet/agatha:redis"
          ports:
            - containerPort: 6379
          env:
            - name: REDIS_IP
              value: $REDIS_IP
            - name: REDIS_PORT
              value: $REDIS_PORT
            - name: REDIS_PASSWORD
              value: $REDIS_PASSWORD

        - image: "gperdrizet/agatha:bot"
          env:
            - name: HOST_IP
              value: agatha_api
            - name: FLASK_PORT
              value: $FLASK_PORT
            - name: TELEGRAM_TOKEN
              value: $TELEGRAM_TOKEN
```

And, go!

```bash
$ gcloud beta run services replace sidecar.yaml

Applying new configuration to Cloud Run service [agatha] in project [ask-agatha] region [us-central1]
X Deploying new service...                                                                                                                                            
  . Creating Revision...                                                                                                                                              
  . Routing traffic...                                                                                                                                                
Deployment failed                                                                                                                                                     
ERROR: (gcloud.beta.run.services.replace) spec.template.spec.containers: Revision template should contain exactly one container with an exposed port.
```

Nope - Ok, I really don't think that Cloud Run is the way to do this. We are spending too much time trying to work around limitations of the service. Agatha is not some simple node.js web app. It's a fully engineered custom feature engineering and inference pipeline. Looks like it's gonna be GKE for us. But it still bothers me that I can't get the flask API up - it's listening. Let's try one more time with tne new image. I'd at least like to see an error complaining that it can't contact redis. At least that way, we would know that we are doing everything right-ish.

## 4. One last shot: Agatha API, stand-alone

```bash
$ gcloud beta run deploy api \
  --image gperdrizet/agatha:api \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-secrets=HF_TOKEN=hf_token:latest \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --port $FLASK_PORT \
  --no-cpu-throttling \
  --cpu 8 \
  --memory 32G \
  --concurrency 2 \
  --max-instances 1 \
  --gpu 1 \
  --gpu-type "nvidia-l4"

Allow unauthenticated invocations to [api] (y/N)?  n

Deploying container to Cloud Run service [api] in project [ask-agatha] region [us-central1]
X Deploying new service...                                                                                                                                            
  - Creating Revision...                                                                                                                                              
  . Routing traffic...                                                                                                                                                
Deployment failed                                                                                                                                                     
ERROR: (gcloud.beta.run.deploy) Revision 'api-00001-wbr' is not ready and cannot serve traffic. The user-provided container failed to start and listen on the port defined provided by the PORT=5000 environment variable within the allocated timeout. This can happen when the container port is misconfigured or if the timeout is too short. The health check timeout can be extended. Logs for this revision might contain more information.

Logs URL: https://console.cloud.google.com/logs/viewer?project=ask-agatha&resource=cloud_run_revision/service_name/api/revision_name/api-00001-wbr&advancedFilter=resource.type%3D%22cloud_run_revision%22%0Aresource.labels.service_name%3D%22api%22%0Aresource.labels.revision_name%3D%22api-00001-wbr%22 
For more troubleshooting guidance, see https://cloud.google.com/run/docs/troubleshooting#container-failed-to-start
```

Looks like it's failing health checks. It's checking the right port though. Let's look at the container logs and see if something else failed inside.

```bash
$ gcloud run services logs read api --limit=100 --project ask-agatha

2024-11-20 21:52:45 ==========
2024-11-20 21:52:45 == CUDA ==
2024-11-20 21:52:45 ==========
2024-11-20 21:52:45 CUDA Version 11.4.3
2024-11-20 21:52:45 Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
2024-11-20 21:52:45 This container image and its contents are governed by the NVIDIA Deep Learning Container License.
2024-11-20 21:52:45 By pulling and using the container, you accept the terms and conditions of this license:
2024-11-20 21:52:45 https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
2024-11-20 21:52:45 A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
2024-11-20 21:52:45 Error: '' is not a valid port number.
```

Oooo, tantalizing - CUDA looks good, but the port number environment variable is empty... Yep, looking at the run command above, I set the port for Cloud Run to talk to the container on, but I didn't pass it in as an environment variable. Trying one more time:

```bash
$ gcloud beta run deploy api \
  --image gperdrizet/agatha:api \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-secrets=HF_TOKEN=hf_token:latest \
  --update-env-vars FLASK_PORT=$FLASK_PORT \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --port $FLASK_PORT \
  --no-cpu-throttling \
  --cpu 8 \
  --memory 32G \
  --concurrency 2 \
  --max-instances 1 \
  --gpu 1 \
  --gpu-type "nvidia-l4"

Deploying container to Cloud Run service [api] in project [ask-agatha] region [us-central1]
✓ Deploying... Done.                                                                                                                                                  
  ✓ Creating Revision...                                                                                                                                              
  ✓ Routing traffic...                                                                                                                                                
Done.                                                                                                                                                                 
Service [api] revision [api-00002-7sl] has been deployed and is serving 100 percent of traffic.
Service URL: https://api-224558092745.us-central1.run.app
```

Woah! We did it. Let's look at the logs:

```bash
$ gcloud run services logs read api --limit=100 --project ask-agatha

2024-11-20 22:10:41 ==========
2024-11-20 22:10:41 == CUDA ==
2024-11-20 22:10:41 ==========
2024-11-20 22:10:42 CUDA Version 11.4.3
2024-11-20 22:10:42 Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
2024-11-20 22:10:42 This container image and its contents are governed by the NVIDIA Deep Learning Container License.
2024-11-20 22:10:42 By pulling and using the container, you accept the terms and conditions of this license:
2024-11-20 22:10:42 https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
2024-11-20 22:10:42 A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
2024-11-20 22:10:44 [2024-11-20 22:10:44 +0000] [38] [INFO] Starting gunicorn 22.0.0
2024-11-20 22:10:44 [2024-11-20 22:10:44 +0000] [38] [INFO] Listening at: http://0.0.0.0:5000 (38)
2024-11-20 22:10:44 [2024-11-20 22:10:44 +0000] [38] [INFO] Using worker: sync
2024-11-20 22:10:44 [2024-11-20 22:10:44 +0000] [40] [INFO] Booting worker with pid: 40
2024-11-20 22:10:52 [2024-11-20 22:10:52 +0000] [40] [ERROR] Exception in worker process
2024-11-20 22:10:52 Traceback (most recent call last):
  File "/usr/local/lib/python3.8/dist-packages/gunicorn/arbiter.py", line 609, in spawn_worker
    worker.init_process()
  File "/usr/local/lib/python3.8/dist-packages/gunicorn/workers/base.py", line 134, in init_process
    self.load_wsgi()
  File "/usr/local/lib/python3.8/dist-packages/gunicorn/workers/base.py", line 146, in load_wsgi
    self.wsgi = self.app.wsgi()
  File "/usr/local/lib/python3.8/dist-packages/gunicorn/app/base.py", line 67, in wsgi
    self.callable = self.load()
  File "/usr/local/lib/python3.8/dist-packages/gunicorn/app/wsgiapp.py", line 58, in load
    return self.load_wsgiapp()
  File "/usr/local/lib/python3.8/dist-packages/gunicorn/app/wsgiapp.py", line 48, in load_wsgiapp
    return util.import_app(self.app_uri)
  File "/usr/local/lib/python3.8/dist-packages/gunicorn/util.py", line 371, in import_app
    mod = importlib.import_module(module)
  File "/usr/lib/python3.8/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1014, in _gcd_import
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 848, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/agatha_api/api.py", line 6, in <module>
    import functions.flask_app as app_funcs
  File "/agatha_api/functions/flask_app.py", line 10, in <module>
    import configuration as config
  File "/agatha_api/configuration.py", line 34, in <module>
    HOST_IP=os.environ['HOST_IP']
  File "/usr/lib/python3.8/os.py", line 675, in __getitem__
    raise KeyError(key) from None
KeyError: 'HOST_IP'
2024-11-20 22:10:52 [2024-11-20 22:10:52 +0000] [40] [INFO] Worker exiting (pid: 40)
2024-11-20 22:10:53 [2024-11-20 22:10:53 +0000] [38] [ERROR] Worker (pid:40) exited with code 3
2024-11-20 22:10:53 [2024-11-20 22:10:53 +0000] [38] [ERROR] Shutting down: Master
2024-11-20 22:10:53 [2024-11-20 22:10:53 +0000] [38] [ERROR] Reason: Worker failed to boot.
```

OK, looks like I forgot to send in the IP for flask to listen on and the redis IP. This is where I think we are going to run into trouble. We don't have a Redis container running. How about this:

```bash
$ gcloud beta run deploy api \
  --image gperdrizet/agatha:api \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-secrets=HF_TOKEN=hf_token:latest \
  --update-env-vars FLASK_PORT=$FLASK_PORT \
  --update-env-vars HOST_IP=$HOST_IP \
  --update-env-vars REDIS_IP=$REDIS_IP \
  --update-env-vars REDIS_PORT=$REDIS_PORT \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --port $FLASK_PORT \
  --no-cpu-throttling \
  --cpu 8 \
  --memory 32G \
  --concurrency 2 \
  --max-instances 1 \
  --gpu 1 \
  --gpu-type "nvidia-l4"
```

```bash
$ gcloud run services logs read api --limit=100 --project ask-agatha

2024-11-20 22:31:46 ==========
2024-11-20 22:31:46 == CUDA ==
2024-11-20 22:31:46 ==========
2024-11-20 22:31:46 CUDA Version 11.4.3
2024-11-20 22:31:46 Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
2024-11-20 22:31:46 This container image and its contents are governed by the NVIDIA Deep Learning Container License.
2024-11-20 22:31:46 By pulling and using the container, you accept the terms and conditions of this license:
2024-11-20 22:31:46 https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
2024-11-20 22:31:46 A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
2024-11-20 22:31:47 [2024-11-20 22:31:47 +0000] [35] [INFO] Starting gunicorn 22.0.0
2024-11-20 22:31:47 [2024-11-20 22:31:47 +0000] [35] [INFO] Listening at: http://0.0.0.0:5000 (35)
2024-11-20 22:31:47 [2024-11-20 22:31:47 +0000] [35] [INFO] Using worker: sync
2024-11-20 22:31:47 [2024-11-20 22:31:47 +0000] [37] [INFO] Booting worker with pid: 37
2024-11-20 22:31:59 [nltk_data] Downloading package stopwords to /root/nltk_data...
2024-11-20 22:31:59 [nltk_data]   Unzipping corpora/stopwords.zip.
2024-11-20 22:31:59 [nltk_data] Downloading package wordnet to /root/nltk_data...
2024-11-20 22:32:00 False
2024-11-20 22:32:00 ===================================BUG REPORT===================================
2024-11-20 22:32:00 ================================================================================
2024-11-20 22:32:00 The following directories listed in your path were found to be non-existent: {PosixPath('/usr/local/nvidia/lib')}
2024-11-20 22:32:00 The following directories listed in your path were found to be non-existent: {PosixPath('gunicorn/22.0.0')}
2024-11-20 22:32:00 CUDA_SETUP: WARNING! libcudart.so not found in any environmental path. Searching in backup paths...
2024-11-20 22:32:00 DEBUG: Possible options found for libcudart.so: {PosixPath('/usr/local/cuda/lib64/libcudart.so.11.0')}
2024-11-20 22:32:00 CUDA SETUP: PyTorch settings found: CUDA_VERSION=117, Highest Compute Capability: 8.9.
2024-11-20 22:32:00 CUDA SETUP: To manually override the PyTorch CUDA version please see:https://github.com/TimDettmers/bitsandbytes/blob/main/how_to_use_nonpytorch_cuda.md
2024-11-20 22:32:00 CUDA SETUP: Required library version not found: libbitsandbytes_cuda117.so. Maybe you need to compile it from source?
2024-11-20 22:32:00 CUDA SETUP: Defaulting to libbitsandbytes_cpu.so...
2024-11-20 22:32:00 ================================================ERROR=====================================
2024-11-20 22:32:00 /usr/local/lib/python3.8/dist-packages/bitsandbytes-0.42.0-py3.8.egg/bitsandbytes/cuda_setup/main.py:167: UserWarning: Welcome to bitsandbytes. For bug reports, please run
2024-11-20 22:32:00 CUDA SETUP: CUDA detection failed! Possible reasons:
2024-11-20 22:32:00 1. You need to manually override the PyTorch CUDA version. Please see: "https://github.com/TimDettmers/bitsandbytes/blob/main/how_to_use_nonpytorch_cuda.md
2024-11-20 22:32:00 2. CUDA driver not installed
2024-11-20 22:32:00 python -m bitsandbytes
2024-11-20 22:32:00 3. CUDA not installed
2024-11-20 22:32:00   warn(msg)
2024-11-20 22:32:00 /usr/local/lib/python3.8/dist-packages/bitsandbytes-0.42.0-py3.8.egg/bitsandbytes/cuda_setup/main.py:167: UserWarning: /usr/local/nvidia/lib:/usr/local/nvidia/lib64 did not contain ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] as expected! Searching further paths...
2024-11-20 22:32:00 4. You have multiple conflicting CUDA libraries
2024-11-20 22:32:00   warn(msg)
2024-11-20 22:32:00 5. Required library not pre-compiled for this bitsandbytes release!
2024-11-20 22:32:00 CUDA SETUP: If you compiled from source, try again with `make CUDA_VERSION=DETECTED_CUDA_VERSION` for example, `make CUDA_VERSION=113`.
2024-11-20 22:32:00 CUDA SETUP: The CUDA version for the compile might depend on your conda install. Inspect CUDA version via `conda list | grep cuda`.
2024-11-20 22:32:00 ================================================================================
2024-11-20 22:32:00 CUDA SETUP: Something unexpected happened. Please compile from source:
2024-11-20 22:32:00 git clone https://github.com/TimDettmers/bitsandbytes.git
2024-11-20 22:32:00 cd bitsandbytes
2024-11-20 22:32:00 CUDA_VERSION=117 make cuda11x
2024-11-20 22:32:00 python setup.py install
2024-11-20 22:32:00 CUDA SETUP: Setup Failed!
2024-11-20 22:32:00 [2024-11-20 22:32:00 +0000] [37] [ERROR] Exception in worker process
2024-11-20 22:32:00 Traceback (most recent call last):
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1778, in _get_module
    return importlib.import_module("." + module_name, self.__name__)
  File "/usr/lib/python3.8/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1014, in _gcd_import
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 848, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/usr/local/lib/python3.8/dist-packages/transformers/integrations/bitsandbytes.py", line 21, in <module>
    import bitsandbytes as bnb
  File "/usr/local/lib/python3.8/dist-packages/bitsandbytes-0.42.0-py3.8.egg/bitsandbytes/__init__.py", line 6, in <module>
    from . import cuda_setup, utils, research
  File "/usr/local/lib/python3.8/dist-packages/bitsandbytes-0.42.0-py3.8.egg/bitsandbytes/research/__init__.py", line 1, in <module>
    from . import nn
  File "/usr/local/lib/python3.8/dist-packages/bitsandbytes-0.42.0-py3.8.egg/bitsandbytes/research/nn/__init__.py", line 1, in <module>
    from .modules import LinearFP8Mixed, LinearFP8Global
  File "/usr/local/lib/python3.8/dist-packages/bitsandbytes-0.42.0-py3.8.egg/bitsandbytes/research/nn/modules.py", line 8, in <module>
    from bitsandbytes.optim import GlobalOptimManager
  File "/usr/local/lib/python3.8/dist-packages/bitsandbytes-0.42.0-py3.8.egg/bitsandbytes/optim/__init__.py", line 6, in <module>
    from bitsandbytes.cextension import COMPILED_WITH_CUDA
  File "/usr/local/lib/python3.8/dist-packages/bitsandbytes-0.42.0-py3.8.egg/bitsandbytes/cextension.py", line 20, in <module>
    raise RuntimeError('''
RuntimeError: 
2024-11-20 22:32:00         CUDA Setup failed despite GPU being available. Please run the following command to get more information:
2024-11-20 22:32:00         python -m bitsandbytes
2024-11-20 22:32:00         Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
2024-11-20 22:32:00         to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
2024-11-20 22:32:00         and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues
2024-11-20 22:32:00 The above exception was the direct cause of the following exception:
2024-11-20 22:32:00 Traceback (most recent call last):
  File "/usr/local/lib/python3.8/dist-packages/gunicorn/arbiter.py", line 609, in spawn_worker
    worker.init_process()
  File "/usr/local/lib/python3.8/dist-packages/gunicorn/workers/base.py", line 134, in init_process
    self.load_wsgi()
  File "/usr/local/lib/python3.8/dist-packages/gunicorn/workers/base.py", line 146, in load_wsgi
    self.wsgi = self.app.wsgi()
  File "/usr/local/lib/python3.8/dist-packages/gunicorn/app/base.py", line 67, in wsgi
    self.callable = self.load()
  File "/usr/local/lib/python3.8/dist-packages/gunicorn/app/wsgiapp.py", line 58, in load
    return self.load_wsgiapp()
  File "/usr/local/lib/python3.8/dist-packages/gunicorn/app/wsgiapp.py", line 48, in load_wsgiapp
    return util.import_app(self.app_uri)
  File "/usr/local/lib/python3.8/dist-packages/gunicorn/util.py", line 371, in import_app
    mod = importlib.import_module(module)
  File "/usr/lib/python3.8/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1014, in _gcd_import
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 848, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/agatha_api/api.py", line 37, in <module>
    reader_model, writer_model=helper_funcs.start_models(logger)
  File "/agatha_api/functions/helper.py", line 179, in start_models
    reader_model.load()
  File "/agatha_api/classes/llm.py", line 74, in load
    self.model = AutoModelForCausalLM.from_pretrained(
  File "/usr/local/lib/python3.8/dist-packages/transformers/models/auto/auto_factory.py", line 564, in from_pretrained
    return model_class.from_pretrained(
  File "/usr/local/lib/python3.8/dist-packages/transformers/modeling_utils.py", line 3657, in from_pretrained
    hf_quantizer.validate_environment(
  File "/usr/local/lib/python3.8/dist-packages/transformers/quantizers/quantizer_bnb_4bit.py", line 78, in validate_environment
    from ..integrations import validate_bnb_backend_availability
  File "<frozen importlib._bootstrap>", line 1039, in _handle_fromlist
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1766, in __getattr__
    module = self._get_module(self._class_to_module[name])
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1780, in _get_module
    raise RuntimeError(
RuntimeError: Failed to import transformers.integrations.bitsandbytes because of the following error (look up to see its traceback):
2024-11-20 22:32:00         CUDA Setup failed despite GPU being available. Please run the following command to get more information:
2024-11-20 22:32:00         python -m bitsandbytes
2024-11-20 22:32:00         Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
2024-11-20 22:32:00         to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
2024-11-20 22:32:00         and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues
2024-11-20 22:32:00 [2024-11-20 22:32:00 +0000] [37] [INFO] Worker exiting (pid: 37)
```

OK, this is actually good news, we very carefully pinned and compiled a specific version of bitsandbytes to work on our older K80 GPUs with. The fact that it has trouble on modern hardware is not a huge surprise. I think the solution there is to clone the GitHub repo to a Compute Engine instance with a L4 GPU and get the container working there. That I can work with.

Let's see if we can get the redis container going.

## 5. One last shot: Redis, stand-alone

```bash
gcloud beta run deploy redis \
  --image gperdrizet/agatha:redis \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-env-vars REDIS_IP=$REDIS_IP \
  --update-env-vars REDIS_PORT=$REDIS_PORT \
  --port $REDIS_PORT \
  --no-cpu-throttling

Allow unauthenticated invocations to [redis] (y/N)?  n

Deploying container to Cloud Run service [redis] in project [ask-agatha] region [us-central1]
✓ Deploying new service... Done.                                                                                                                                                              
  ✓ Creating Revision...                                                                                                                                                                      
  ✓ Routing traffic...                                                                                                                                                                        
Done.                                                                                                                                                                                         
Service [redis] revision [redis-00001-bvn] has been deployed and is serving 100 percent of traffic.
Service URL: https://redis-224558092745.us-central1.run.app
```

Holy cow! Working too - check the container logs:

```bash
$ gcloud run services logs read redis --limit=100 --project ask-agatha

2024-11-21 04:30:57 6379
2024-11-21 04:30:57 0.0.0.0
2024-11-21 04:30:57 sysctl: write error: I/O error
2024-11-21 04:30:57 3:C 21 Nov 2024 04:30:57.624 # WARNING Memory overcommit must be enabled! Without it, a background save or replication may fail under low memory condition. Being disabled, it can also cause failures without low memory condition, see https://github.com/jemalloc/jemalloc/issues/1328. To fix this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory=1' for this to take effect.
2024-11-21 04:30:57 3:M 21 Nov 2024 04:30:57.627 # WARNING: The TCP backlog setting of 511 cannot be enforced because /proc/sys/net/core/somaxconn is set to the lower value of 128.
2024-11-21 04:34:10 GET 403 https://redis-224558092745.us-central1.run.app/
2024-11-21 04:34:10 GET 403 https://redis-224558092745.us-central1.run.app/favicon.ico
```

OK, good. Still have that pesky overcommit error, let's not worry about it for now.

Last one is the bot.

## 6. Telegram bot

Issue here is, we know this one will fail the health check because it is not running a webserver or listening on a port at all. I can think of three options:

1. Add a simple flask server just for the health check
2. Run it as a job so that there is no health check
3. Run it as a sidecar to the API

The last one seems like the most elegant, 'correct' solution. Let's give it a shot. Tried with a sidecar.yaml file, but couldn't figure out how to pass secrets from secrets manager into the container as environment variables. According to [the documentation](https://cloud.google.com/run/docs/deploying#gcloud_2) it's also possible to run a sidecar configuration via `gcloud run deploy`. Let's try that.

```bash
gcloud run deploy agatha \
  --container api \
  --image=gperdrizet/agatha:api \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-secrets=HF_TOKEN=hf_token:latest \
  --set-env-vars=FLASK_PORT=$FLASK_PORT \
  --set-env-vars=HOST_IP=$HOST_IP \
  --set-env-vars=REDIS_IP=$REDIS_IP \
  --set-env-vars=REDIS_PORT=$REDIS_PORT \
  --service-account=ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --port=$FLASK_PORT \
  --no-cpu-throttling \
  --cpu=8 \
  --memory=32G \
  --concurrency=2 \
  --max-instances=1 \
  --gpu=1 \
  --gpu-type="nvidia-l4" \
  --container=bot \
  --image=gperdrizet/agatha:api \
  --update-secrets=TELEGRAM_TOKEN=telegram_token:latest \
  --set-env-vars=HOST_IP=$HOST_IP \
  --set-env-vars=FLASK_PORT=$FLASK_PORT \
  --service-account=ask-agatha-service@ask-agatha.iam.gserviceaccount.com
```

OK, getting errors about different release tracks. Seems like arguments from the beta standard? release tracks are not compatible. We need the beta track to run the API container with the GPU. So what if we make the bot a sidecar to the redis container and the API stand-alone?

```bash
gcloud beta run deploy agatha \
  --container redis \
  --image gperdrizet/agatha:redis \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-env-vars REDIS_IP=$REDIS_IP \
  --update-env-vars REDIS_PORT=$REDIS_PORT \
  --port $REDIS_PORT \
  --container=bot \
  --image=gperdrizet/agatha:bot \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --update-secrets=TELEGRAM_TOKEN=telegram_token:latest \
  --set-env-vars=HOST_IP=$HOST_IP \
  --set-env-vars=FLASK_PORT=$FLASK_PORT

ERROR: (gcloud.beta.run.deploy) unrecognized arguments:
  
 --service-account flag is available in one or more alternate release tracks. Try:

  gcloud run deploy --service-account
  gcloud alpha run deploy --service-account

  --service-account (did you mean '--service-account'?)
  ask-agatha-service@ask-agatha.iam.gserviceaccount.com
  To search the help text of gcloud commands, run:
  gcloud help -- SEARCH_TERMS
```

Going around in circles. The above error makes me thing that I should not use the beta release channel, so:

```bash
$ gcloud run deploy agatha \
  --container redis \
  --image=gperdrizet/agatha:redis \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-env-vars REDIS_IP=$REDIS_IP \
  --update-env-vars REDIS_PORT=$REDIS_PORT \
  --port $REDIS_PORT \
  --container bot \
  --image=gperdrizet/agatha:bot \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --update-secrets=TELEGRAM_TOKEN=telegram_token:latest \
  --set-env-vars=HOST_IP=$HOST_IP \
  --set-env-vars=FLASK_PORT=$FLASK_PORT

ERROR: (gcloud.run.deploy) unrecognized arguments:
  
 --service-account flag is available in one or more alternate release tracks. Try:

  gcloud alpha run deploy --service-account
  gcloud beta run deploy --service-account

  --service-account (did you mean '--service-account'?)
  ask-agatha-service@ask-agatha.iam.gserviceaccount.com
  To search the help text of gcloud commands, run:
  gcloud help -- SEARCH_TERMS
```

So, which is it? And without the service account, we can't use the secrets manager... Wait, I figured it out. The --service-account flag must come before the first --container flag. Like this:

```bash
$ gcloud run deploy agatha \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --container redis \
  --image=gperdrizet/agatha:redis \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-env-vars REDIS_IP=$REDIS_IP \
  --update-env-vars REDIS_PORT=$REDIS_PORT \
  --port $REDIS_PORT \
  --container bot \
  --image=gperdrizet/agatha:bot \
  --update-secrets=TELEGRAM_TOKEN=telegram_token:latest \
  --set-env-vars=HOST_IP=$HOST_IP \
  --set-env-vars=FLASK_PORT=$FLASK_PORT

Deploying container to Cloud Run service [agatha] in project [ask-agatha] region [us-central1]
✓ Deploying... Done.                                                                                                 
  ✓ Creating Revision...                                                                                             
  ✓ Routing traffic...                                                                                               
Done.                                                                                                                
Service [agatha] revision [agatha-00003-fkx] has been deployed and is serving 100 percent of traffic.
Service URL: https://agatha-224558092745.us-central1.run.app
```

OK, almost there, let's set it up so that redis is a standalone container and the bot runs as a sidecar to the API.

```bash
$ gcloud beta run deploy agatha \
  --service-account=ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --concurrency=2 \
  --max-instances=1 \
  --no-cpu-throttling \
  --gpu-type="nvidia-l4" \
  --container api \
  --image=gperdrizet/agatha:api \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-secrets=HF_TOKEN=hf_token:latest \
  --set-env-vars=FLASK_PORT=$FLASK_PORT \
  --set-env-vars=HOST_IP=$HOST_IP \
  --set-env-vars=REDIS_IP=$REDIS_IP \
  --set-env-vars=REDIS_PORT=$REDIS_PORT \
  --port=$FLASK_PORT \
  --cpu=4 \
  --memory=16Gi \
  --gpu=1 \
  --container=bot \
  --image=gperdrizet/agatha:bot \
  --update-secrets=TELEGRAM_TOKEN=telegram_token:latest \
  --set-env-vars=HOST_IP=$HOST_IP \
  --set-env-vars=FLASK_PORT=$FLASK_PORT

Allow unauthenticated invocations to [agatha] (y/N)?  n

Deploying container to Cloud Run service [agatha] in project [ask-agatha] region [us-central1]
✓ Deploying new service... Done.                                                                                     
  ✓ Creating Revision...                                                                                             
  ✓ Routing traffic...                                                                                               
Done.                                                                                                                
Service [agatha] revision [agatha-00001-lms] has been deployed and is serving 100 percent of traffic.
Service URL: https://agatha-224558092745.us-central1.run.app
```

Order of arguments really matters here. Also, setting `--cpu 8` gives the error: `ERROR: (gcloud.beta.run.deploy) spec.template.spec.containers.resources.limits.cpu: Invalid value specified for cpu. Total millicpu may not exceed 8000.` Also, GPU needs at least 16Gi memory - that's Gi, not G... don't get it wrong.

Last, redis:

```bash
$ gcloud run deploy redis \
  --image gperdrizet/agatha:redis \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-env-vars REDIS_IP=$REDIS_IP \
  --update-env-vars REDIS_PORT=$REDIS_PORT \
  --port $REDIS_PORT \
  --no-cpu-throttling

Allow unauthenticated invocations to [redis] (y/N)?  n

Deploying container to Cloud Run service [redis] in project [ask-agatha] region [us-central1]
✓ Deploying new service... Done.                                                                                     
  ✓ Creating Revision...                                                                                             
  ✓ Routing traffic...                                                                                               
Done.                                                                                                                
Service [redis] revision [redis-00001-8x8] has been deployed and is serving 100 percent of traffic.
Service URL: https://redis-224558092745.us-central1.run.app
```

OK! Calling this a deployment? Still have issues:

1. There are problems with bitsandbytes inside of the API container - solution is probably to just pip install it rather than go through all of the trouble of compiling a specific version for compatibility with k80 cards.
2. Networking - solution is to read more.

Modified API container to install bitsandbytes via pip and re-built, pushed to Docker hub.

## 7. Secrets

After accidentally committing some secrets in this file, update everything. Had to delete secrets manually from the Cloud Consol Secret Manager and re-create:

```bash
printf $REDIS_PASSWORD | gcloud secrets create redis_password --data-file=- --replication-policy="automatic"
printf $HF_TOKEN | gcloud secrets create hf_token --data-file=- --replication-policy="automatic"
printf $TELEGRAM_TOKEN | gcloud secrets create telegram_token --data-file=- --replication-policy="automatic"
```

Re-deploy everything and check the logs:

```bash
$ gcloud run services logs read agatha --limit=100 --project ask-agatha

2024-11-21 12:26:47 ==========
2024-11-21 12:26:47 == CUDA ==
2024-11-21 12:26:47 ==========
2024-11-21 12:26:47 CUDA Version 11.4.3
2024-11-21 12:26:47 Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
2024-11-21 12:26:47 This container image and its contents are governed by the NVIDIA Deep Learning Container License.
2024-11-21 12:26:47 By pulling and using the container, you accept the terms and conditions of this license:
2024-11-21 12:26:47 https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
2024-11-21 12:26:47 A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
2024-11-21 12:26:49 [2024-11-21 12:26:49 +0000] [36] [INFO] Starting gunicorn 23.0.0
2024-11-21 12:26:49 [2024-11-21 12:26:49 +0000] [36] [INFO] Listening at: http://0.0.0.0:5000 (36)
2024-11-21 12:26:49 [2024-11-21 12:26:49 +0000] [36] [INFO] Using worker: sync
2024-11-21 12:26:49 [2024-11-21 12:26:49 +0000] [38] [INFO] Booting worker with pid: 38
2024-11-21 12:26:50 Will log to: /agatha_bot/logs/telegram_bot.log
2024-11-21 12:27:01 [nltk_data] Downloading package stopwords to /root/nltk_data...
2024-11-21 12:27:01 [nltk_data]   Unzipping corpora/stopwords.zip.
2024-11-21 12:27:01 [nltk_data] Downloading package wordnet to /root/nltk_data...
2024-11-21 12:29:06 Downloading shards:   0%|          | 0/4 [00:00<?, ?it/s]
2024-11-21 12:31:05 Downloading shards:  25%|██▌       | 1/4 [01:58<05:56, 118.77s/it]
2024-11-21 12:32:04 Downloading shards:  50%|█████     | 2/4 [03:58<03:58, 119.06s/it]
```

Sweet, looks like we got it - all we need now is the networking!

## 8. Networking

Now, we need communication between the containers. From the documentation, it sounds like we need to set-up a virtual private cloud network (VPC). Let's read up on that. Two sources:

1. [VPC networks](https://cloud.google.com/vpc/docs/vpc)
2. [Private networking and Cloud Run](https://cloud.google.com/run/docs/securing/private-networking)

### 8.1. VPC networks

Internal Google Cloud network for all of your stuff:

- Connectivity for VMs
- Proxies/load balancers
- VPN/VLAN connections to external machines (pyrite!)
- Sends traffic from external load balancers to services

Networks are global subnets are zonal.

Can create multiple networks in auto or custom mode:

1. Auto: makes on subnet in each region using per-defined IP range. Can add more subnets manualy.
2. Custom: make the subnets yourself.

Projects start with a default auto mode network with IPv4 firewall rules in place, but no IPv6 rules.

Had to enable compute engine API to see VPC network page in Cloud Console.

Sounds like we need to set up direct VPC egress for our source services. The docs also mention making sure that traffic going to Cloud Run routes through the VPC - I don't know if we need to worry about this because nothing needs to call in to our services yet. The bot just calls out. As long as the containers can talk to each other and the bot container has internet access we are golden.

#### 8.1. Configure services for direct VPC egress

Documentation [here](https://cloud.google.com/run/docs/configuring/vpc-direct-vpc#direct-vpc-service).

Looks like we need to add the following to our deploy commands:

```text
--network=NETWORK
--subnet=SUBNET
--network-tags=NETWORK_TAG_NAMES
--vpc-egress=EGRESS_SETTING
--region=REGION
```

OK, let's figure out the values:

1. network: name of VPC network: 'default' from Cloud Console networking page
2. subnet: name of subnet: also 'default' from Cloud Console network page, 'SUBNETS' tab
3. network-tags: optional, tags for service revisions
4. vpc-egress: egress setting value, either 'all-traffic' or 'private-ranges-only'
5. region: services region: us-cental1

OK, try it!

```bash
$ gcloud run deploy redis \
  --image gperdrizet/agatha:redis \
  --service-account ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-env-vars REDIS_IP=$REDIS_IP \
  --update-env-vars REDIS_PORT=$REDIS_PORT \
  --port $REDIS_PORT \
  --no-cpu-throttling \
  --network=default \
  --subnet=default \
  --vpc-egress=all-traffic \
  --region=us-central1

Deploying container to Cloud Run service [redis] in project [ask-agatha] region [us-central1]
✓ Deploying... Done.                                                                         
  ✓ Creating Revision...                                                                     
  ✓ Routing traffic...                                                                       
Done.                                                                                        
Service [redis] revision [redis-00002-xvs] has been deployed and is serving 100 percent of traffic.
Service URL: https://redis-224558092745.us-central1.run.app
```

```bash
$ gcloud beta run deploy agatha \
  --service-account=ask-agatha-service@ask-agatha.iam.gserviceaccount.com \
  --concurrency=2 \
  --max-instances=1 \
  --no-cpu-throttling \
  --gpu-type="nvidia-l4" \
  --network=default \
  --subnet=default \
  --vpc-egress=all-traffic \
  --region=us-central1 \
  --container api \
  --image=gperdrizet/agatha:api \
  --update-secrets=REDIS_PASSWORD=redis_password:latest \
  --update-secrets=HF_TOKEN=hf_token:latest \
  --set-env-vars=FLASK_PORT=$FLASK_PORT \
  --set-env-vars=HOST_IP=$HOST_IP \
  --set-env-vars=REDIS_IP=$REDIS_IP \
  --set-env-vars=REDIS_PORT=$REDIS_PORT \
  --port=$FLASK_PORT \
  --cpu=4 \
  --memory=16Gi \
  --gpu=1 \
  --container=bot \
  --image=gperdrizet/agatha:bot \
  --update-secrets=TELEGRAM_TOKEN=telegram_token:latest \
  --set-env-vars=HOST_IP=$HOST_IP \
  --set-env-vars=FLASK_PORT=$FLASK_PORT

Service URL: https://redis-224558092745.us-central1.run.app
Deploying container to Cloud Run service [agatha] in project [ask-agatha] region [us-central1]
✓ Deploying... Done.                                                                         
  ✓ Creating Revision...                                                                     
  ✓ Routing traffic...                                                                       
Done.                                                                                        
Service [agatha] revision [agatha-00002-kbz] has been deployed and is serving 100 percent of traffic.
Service URL: https://agatha-224558092745.us-central1.run.app
```

Sweet, no obvious errors. Now check the networking configuration with:

```bash
$ gcloud run services describe redis --region=us-central1

✔ Service redis in region us-central1
 
URL:     https://redis-224558092745.us-central1.run.app
Ingress: all
Traffic:
  100% LATEST (currently redis-00002-xvs)
 
Last updated on 2024-11-21T15:24:03.170653Z by gperdrizet@ask-agatha.com:
  Revision redis-00002-xvs
  Container None
    Image:           gperdrizet/agatha:redis
    Port:            6379
    Memory:          512Mi
    CPU:             1000m
    Env vars:
      REDIS_IP       0.0.0.0
      REDIS_PORT     6379
    Secrets:
      REDIS_PASSWORD redis_password:latest
    Startup Probe:
      TCP every 240s
      Port:          6379
      Initial delay: 0s
      Timeout:       240s
      Failure threshold: 1
      Type:          Default
  Service account:   ask-agatha-service@ask-agatha.iam.gserviceaccount.com
  Concurrency:       80
  Max Instances:     100
  Timeout:           300s
  VPC access:
    Network:         default
    Subnet:          default
    Egress:          all-traffic
  CPU Allocation:    CPU is always allocated
```

OK, VPC access stanza looks good!

```bash
$ gcloud run services describe agatha --region=us-central1

URL:     https://agatha-224558092745.us-central1.run.app
Ingress: all
Traffic:
  100% LATEST (currently agatha-00002-kbz)
 
Last updated on 2024-11-21T15:24:41.735267Z by gperdrizet@ask-agatha.com:
  Revision agatha-00002-kbz
  Container api
    Image:           gperdrizet/agatha:api
    Port:            5000
    Memory:          16Gi
    CPU:             4
    GPU:             1
    GPU Type:        nvidia-l4
    Env vars:
      FLASK_PORT     5000
      HOST_IP        0.0.0.0
      REDIS_IP       0.0.0.0
      REDIS_PORT     6379
    Secrets:
      HF_TOKEN       hf_token:latest
      REDIS_PASSWORD redis_password:latest
    Startup Probe:
      TCP every 240s
      Port:          5000
      Initial delay: 0s
      Timeout:       240s
      Failure threshold: 1
      Type:          Default
  Container bot
    Image:           gperdrizet/agatha:bot
    Memory:          256Mi
    CPU:             1000m
    Env vars:
      FLASK_PORT     5000
      HOST_IP        0.0.0.0
    Secrets:
      TELEGRAM_TOKEN telegram_token:latest
  Service account:   ask-agatha-service@ask-agatha.iam.gserviceaccount.com
  Concurrency:       2
  Max Instances:     1
  Timeout:           300s
  VPC access:
    Network:         default
    Subnet:          default
    Egress:          all-traffic
  CPU Allocation:    CPU is always allocated
```

Looks good! But how do I know what IPs to use? I'm assuming 0.0.0.0 is not right, that's fine for the service inside the container, but what is the service container's IP? Did this with container names via a network set-up via compose.yaml locally...

Maybe we use the urls it provides? Let's try that I guess.

- Redis: https://redis-224558092745.us-central1.run.app
- Agatha API: https://agatha-224558092745.us-central1.run.app

Now, tell the API the redis ip is the redis url and tell the bot that the API ip is the API url. 

Also, check the firewall rules. From Cloud Consol go to Networking > VPC network > FIREWALLS. It looks like all of the rules relate to ingress. We might be OK, since we don't need to call into the VPC. Let's try it!

OK, no obvious errors!... Agatha, are you there?

```bash
$ gcloud run services logs read agatha --limit=100 --project ask-agatha

2024-11-21 15:41:05 ==========
2024-11-21 15:41:05 == CUDA ==
2024-11-21 15:41:05 ==========
2024-11-21 15:41:05 CUDA Version 11.4.3
2024-11-21 15:41:05 Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
2024-11-21 15:41:05 This container image and its contents are governed by the NVIDIA Deep Learning Container License.
2024-11-21 15:41:05 By pulling and using the container, you accept the terms and conditions of this license:
2024-11-21 15:41:05 https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
2024-11-21 15:41:05 A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
2024-11-21 15:41:07 Will log to: /agatha_bot/logs/telegram_bot.log
2024-11-21 15:41:07 [2024-11-21 15:41:07 +0000] [35] [INFO] Starting gunicorn 23.0.0
2024-11-21 15:41:07 [2024-11-21 15:41:07 +0000] [35] [INFO] Listening at: http://0.0.0.0:5000 (35)
2024-11-21 15:41:07 [2024-11-21 15:41:07 +0000] [35] [INFO] Using worker: sync
2024-11-21 15:41:07 [2024-11-21 15:41:07 +0000] [37] [INFO] Booting worker with pid: 37
2024-11-21 15:41:13 Traceback (most recent call last):
  File "/usr/local/lib/python3.10/site-packages/httpx/_transports/default.py", line 72, in map_httpcore_exceptions
    yield
  File "/usr/local/lib/python3.10/site-packages/httpx/_transports/default.py", line 377, in handle_async_request
    resp = await self._pool.handle_async_request(req)
  File "/usr/local/lib/python3.10/site-packages/httpcore/_async/connection_pool.py", line 216, in handle_async_request
    raise exc from None
  File "/usr/local/lib/python3.10/site-packages/httpcore/_async/connection_pool.py", line 196, in handle_async_request
    response = await connection.handle_async_request(
  File "/usr/local/lib/python3.10/site-packages/httpcore/_async/connection.py", line 99, in handle_async_request
    raise exc
  File "/usr/local/lib/python3.10/site-packages/httpcore/_async/connection.py", line 76, in handle_async_request
    stream = await self._connect(request)
  File "/usr/local/lib/python3.10/site-packages/httpcore/_async/connection.py", line 122, in _connect
    stream = await self._network_backend.connect_tcp(**kwargs)
  File "/usr/local/lib/python3.10/site-packages/httpcore/_backends/auto.py", line 30, in connect_tcp
    return await self._backend.connect_tcp(
  File "/usr/local/lib/python3.10/site-packages/httpcore/_backends/anyio.py", line 115, in connect_tcp
    with map_exceptions(exc_map):
  File "/usr/local/lib/python3.10/contextlib.py", line 153, in __exit__
    self.gen.throw(typ, value, traceback)
  File "/usr/local/lib/python3.10/site-packages/httpcore/_exceptions.py", line 14, in map_exceptions
    raise to_exc(exc) from exc
2024-11-21 15:41:13 httpcore.ConnectTimeout
2024-11-21 15:41:13 The above exception was the direct cause of the following exception:
2024-11-21 15:41:13 Traceback (most recent call last):
  File "/usr/local/lib/python3.10/site-packages/telegram/request/_httpxrequest.py", line 293, in do_request
    res = await self._client.request(
  File "/usr/local/lib/python3.10/site-packages/httpx/_client.py", line 1585, in request
    return await self.send(request, auth=auth, follow_redirects=follow_redirects)
  File "/usr/local/lib/python3.10/site-packages/httpx/_client.py", line 1674, in send
    response = await self._send_handling_auth(
  File "/usr/local/lib/python3.10/site-packages/httpx/_client.py", line 1702, in _send_handling_auth
    response = await self._send_handling_redirects(
  File "/usr/local/lib/python3.10/site-packages/httpx/_client.py", line 1739, in _send_handling_redirects
    response = await self._send_single_request(request)
  File "/usr/local/lib/python3.10/site-packages/httpx/_client.py", line 1776, in _send_single_request
    response = await transport.handle_async_request(request)
  File "/usr/local/lib/python3.10/site-packages/httpx/_transports/default.py", line 376, in handle_async_request
    with map_httpcore_exceptions():
  File "/usr/local/lib/python3.10/contextlib.py", line 153, in __exit__
    self.gen.throw(typ, value, traceback)
  File "/usr/local/lib/python3.10/site-packages/httpx/_transports/default.py", line 89, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
2024-11-21 15:41:13 httpx.ConnectTimeout
2024-11-21 15:41:13 The above exception was the direct cause of the following exception:
2024-11-21 15:41:13 Traceback (most recent call last):
  File "/agatha_bot/./bot.py", line 159, in <module>
    application.run_polling()
  File "/usr/local/lib/python3.10/site-packages/telegram/ext/_application.py", line 865, in run_polling
    return self.__run(
  File "/usr/local/lib/python3.10/site-packages/telegram/ext/_application.py", line 1063, in __run
    loop.run_until_complete(self.initialize())
  File "/usr/local/lib/python3.10/asyncio/base_events.py", line 649, in run_until_complete
    return future.result()
  File "/usr/local/lib/python3.10/site-packages/telegram/ext/_application.py", line 487, in initialize
    await self.bot.initialize()
  File "/usr/local/lib/python3.10/site-packages/telegram/ext/_extbot.py", line 297, in initialize
    await super().initialize()
  File "/usr/local/lib/python3.10/site-packages/telegram/_bot.py", line 761, in initialize
    await self.get_me()
  File "/usr/local/lib/python3.10/site-packages/telegram/ext/_extbot.py", line 1920, in get_me
    return await super().get_me(
  File "/usr/local/lib/python3.10/site-packages/telegram/_bot.py", line 893, in get_me
    result = await self._post(
  File "/usr/local/lib/python3.10/site-packages/telegram/_bot.py", line 617, in _post
    return await self._do_post(
  File "/usr/local/lib/python3.10/site-packages/telegram/ext/_extbot.py", line 351, in _do_post
    return await super()._do_post(
  File "/usr/local/lib/python3.10/site-packages/telegram/_bot.py", line 646, in _do_post
    result = await request.post(
  File "/usr/local/lib/python3.10/site-packages/telegram/request/_baserequest.py", line 202, in post
    result = await self._request_wrapper(
  File "/usr/local/lib/python3.10/site-packages/telegram/request/_baserequest.py", line 334, in _request_wrapper
    code, payload = await self.do_request(
  File "/usr/local/lib/python3.10/site-packages/telegram/request/_httpxrequest.py", line 310, in do_request
    raise TimedOut from err
telegram.error.TimedOut: Timed out
```

OK, think I figured it out - line 159 in bot.py is where we start the application and begin polling the Telegram servers for updates. Seems like somehow in out VPC network config we blocked access to the outside. Seem to have fixed it by setting `--vpc-egress=private-ranges-only`.

Ok, no errors or obvious issues, but no answer on the Telegram app. I wish I could see my nice logs - but I can't find a way to access the filesystem of a running container. Let's update the services to log to STDOUT.

OK, added the following to the `start_logger()` helper function for the API and the bot:

```python
stdout_handler=logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)
```

Seems to work locally - rebuild and push the images.

Ok, new issue - it now seems like the API container may be crashing or something. I can see the network/CPU/memory utilization rise at the models are being downloaded, but they never seemed to finish. memory goes to about 50% and bytes received from the internet goes up to 40 M/s second for a minute or two and then everything stops. Like, even the traces stop. But the weird thing is the container is healthy in the Cloud console. Not sure what is going on. Are we reading the models into memory? We do only have 16 Gi, but the API doesn't use that much system memory...

OK, making progress - you need to set 8 CPUs to max out the memory at 32 Gi for a container, but for sidecar configs, the CPUs are split between the containers, so gave the API 6 CPUs with 24 Gi memory and the bot 2 CPUs. Let's see if we can download the models now.

Nope, same behavior. I am starting to think maybe it's being spun down because Google see it is alive and healthy, but it's not processing requests. We did remove the `--no-cpu-throttling` flag. We could try putting it back, but I think it was in conflict with other parts of the configuration. The other option is to put the models in the container. This is more Cloud Run crap that makes me think we really should build this on GKE.
