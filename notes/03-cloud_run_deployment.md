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

### 1. Setup

- Sign in to Google Cloud
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

##### 2.2.1. Text classification API

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

##### 2.2.2. Troubleshooting

OK, so could be any number of problems:

1. The API container image is base on nvidia/cuda:11.4.3-runtime-ubuntu20.04, so it could be the fact that we don't have a GPU.
2. On first spin-up we download the models, which could be exceeding the startup timeout.
3. There could be other problems inside the container, for example, the redis server is not running....

Let's see if we can get anything useful from the logs - nope not, really. Would love to see the logs from inside the container. Not sure how to do that. Let's start doing things we know we need...

##### 2.2.3. Attach GPU(s)

Following the [GPU (services) documentation](https://cloud.google.com/run/docs/configuring/services/gpu).

###### 2.2.3.1. Configure CPU always allocated

```bash
gcloud run services update api --no-cpu-throttling
```

**Note**: can also be set during deployment with by passing `--no-cpu-throttling`.

###### 2.2.3.2. Configure 8 CPUs

```bash
gcloud run services update api --cpu 8
```

**Note**: can also be set during deployment with by passing `--cpu`.

###### 2.2.3.3. Configure 32 GB memory

```bash
gcloud run services update api --memory 32G
```

**Note**: can also be set during deployment with by passing `--memory`.

###### 2.2.3.4. Configure concurrency

Let's set for 2, that's the most workers we can handle on a single GPU.

```bash
gcloud run services update api --concurrency 2
```

**Note**: can also be set during deployment with by passing `--concurrency`.

###### 2.2.3.5. Configure max instances

```bash
gcloud run services update api --max-instances 1
```

**Note**: can also be set during deployment with by passing `--max-instances`.

###### 2.2.3.6. Deploy with GPU

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

##### 2.2.4. Redis container

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

##### start_server.sh

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

Nope - no help. Exact same issue. Don't think `EXPOSE` is sufficent. If I were going to run this container localy with docker I would add `-p 6379:6379` to map the port - do I need to do that somehow with glcoud?


