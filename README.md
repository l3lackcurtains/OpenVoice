#### Build the docker image and push to gcr

```bash
gcloud auth configure-docker
docker login gcr.io

docker build -t silk-open-voice .
docker tag silk-open-voice gcr.io/citric-lead-450721-v2/silk-open-voice:1.0.0
docker push gcr.io/citric-lead-450721-v2/silk-open-voice:1.0.0


```