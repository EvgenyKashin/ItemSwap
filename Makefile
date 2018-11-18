IMAGE=digitman/pytorch_tf_gpu

DOCKER_BUILD=docker build -t ${IMAGE} -f Dockerfile.gpu .
DOCKER_RUN=docker run --rm -it -v ${CURDIR}:/app ${IMAGE}

docker-build:
	${DOCKER_BUILD}

docker-push:
	docker push ${IMAGE}

run-bash:
	${DOCKER_RUN} /bin/bash

run-jupyter:
	docker run --rm -it -v ${CURDIR}:/app -w /app -p 8892:8892 ${IMAGE} jupyter notebook \
		--ip=0.0.0.0 --port=8892 --no-browser --allow-root \
		--NotebookApp.token='' --NotebookApp.password=''

load-prepare-data:
	${DOCKER_RUN} python load_prepare_data.py