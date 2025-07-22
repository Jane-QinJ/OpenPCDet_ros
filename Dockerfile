FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
LABEL maintainer="Jane QIN <1139804380@qq.com>"
# Just in case we need it
ENV DEBIAN_FRONTEND noninteractive

RUN apt update && apt install -y --no-install-recommends git curl wget git zsh tmux vim g++
# needs to be done before we can apply the patches
RUN git config --global user.email "1139804380@qq.com"
RUN git config --global user.name "Jane-QINJ"

# install zsh
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- \
    -t robbyrussell \
    -p git \
    -p ssh-agent \
    -p https://github.com/agkozak/zsh-z \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting


# ==========> INSTALL ROS noetic <=============
RUN apt update && apt install -y curl lsb-release
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt update && apt install -y ros-noetic-desktop-full
RUN apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential -y \
    && rosdep init && rosdep update
RUN echo "source /opt/ros/noetic/setup.zsh" >> ~/.zshrc
RUN echo "source /opt/ros/noetic/setup.bashrc" >> ~/.bashrc

# =========> INSTALL OpenPCDet <=============
RUN apt update && apt install -y python3-pip
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install spconv-cu113
RUN apt update && apt install -y python3-setuptools
# RUN mkdir -p /home/kin/workspace
# WORKDIR /home/firo/Documents/workspace/ # need run inside the container!!!
# RUN git clone https://github.com/Kin-Zhang/OpenPCDet.git 
# RUN cd OpenPCDet && pip3 install -r requirements.txt # need run inside the container!!!
RUN pip3 install pyquaternion numpy==1.23 pillow==8.4 mayavi open3d
# RUN cd OpenPCDet && python3 setup.py develop # need run inside the container!!!

# =========> Clone ROS Package <============
RUN apt update && apt install ros-noetic-ros-numpy ros-noetic-vision-msgs
RUN git clone https://github.com/Kin-Zhang/OpenPCDet_ros.git /home/kin/workspace/OpenPCDet_ws/src/OpenPCDet_ros
RUN apt-get install -y ros-noetic-catkin python3-catkin-tools


# docker run
# docker run --gpus all --rm -d -ti -v /home/$USER/:/home/$USER/ --gpus all -e DISPLAY --net=host --ipc host --name pcdet_ros openpcdetros:latest /bin/zsh
# docker exec -it pcdet_ros /bin/zsh

# inside OpenPCDet_ws/src/OpenPCDet_ros
# catkin build
# source devel/setup.zsh
# roslaunch openpcdet_3d_object_detection.launch