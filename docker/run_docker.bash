docker run --rm -it --privileged --net=host --ipc=host \
-v $HOME/.Xauthority:/home/$(id -un)/.Xauthority \
-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
-e XAUTHORITY=/home/$(id -un)/.Xauthority \
-v $HOME/cnn_bert:/home/digital/cnn_bert \
-e DOCKER_USER_NAME=$(id -un) \
-e DOCKER_USER_ID=$(id -u) \
-e DOCKER_USER_GROUP_NAME=$(id -gn) \
-e DOCKER_USER_GROUP_ID=$(id -g) \
syuntoku/pedsim_ros:melodic_cpu bash -c "echo 'export PATH=/home/$(id -un)/miniconda/bin:${$+$}PATH' >> /home/$(id -un)/.zshrc && terminator"
# --device=/dev/dri:/dev/dri \
