FROM python

RUN mkdir /workspace
WORKDIR /workspace 

RUN pip install torch lightning wandb transformers datasets tqdm numpy nltk pillow sentencepiece

RUN apt update
RUN apt install vim -y

RUN apt install -y cmake build-essential libprotobuf-dev protobuf-compiler

RUN cd /workspace

CMD ["bash", "-c", "wandb login && exec bash"]