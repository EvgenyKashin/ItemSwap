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

load-prepare-face-data:
	${DOCKER_RUN} python load_prepare_face_data.py

clear-data:
	rm data_faces/videos/* -rf && rm data_faces/binary_masks -rf && rm data_faces/facesA -rf \
		&& rm data_faces/facesB -rf && rm data_faces/null.mp4 \
		&& rm weights_faces/gan_models -rf