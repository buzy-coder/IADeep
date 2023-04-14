FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel
WORKDIR /workspace
RUN mkdir workloads
RUN mkdir etcd_key
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN apt-get install -y vim && apt-get install -y libglib2.0-dev && apt-get install -y libglib2.0-0
ADD torch-1.9.0+cu111-cp37-cp37m-linux_x86_64.whl .
RUN pip install torch-1.9.0+cu111-cp37-cp37m-linux_x86_64.whl
RUN rm torch-1.9.0+cu111-cp37-cp37m-linux_x86_64.whl
# RUN apt-get install -y openssh-server
RUN apt-get install -y libgl1-mesa-glx
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip install torch-sparse==0.6.10 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip install torch-geometric
# RUN pip install torch-scatter torch-sparse==0.6.12 torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
COPY requirements.txt /workspace
RUN pip3 install -r requirements.txt
ADD torchvision-0.10.0+cu111-cp37-cp37m-linux_x86_64.whl .
RUN pip uninstall torchvision -y
RUN pip install torchvision-0.10.0+cu111-cp37-cp37m-linux_x86_64.whl
RUN rm torchvision-0.10.0+cu111-cp37-cp37m-linux_x86_64.whl
COPY workloads /workspace/workloads/
COPY etcd_key /workspace/etcd_key/
# Let bert can print to kubectl logs
ENV PYTHONUNBUFFERED=1
ENV ETCD_SERVER_IP="10.119.46.42"
ENV ETCD_PORT=2379
