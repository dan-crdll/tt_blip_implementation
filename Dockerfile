FROM python

RUN mkdir /workspace
WORKDIR /workspace 

COPY . /workspace/

RUN pip install torch lightning wandb transformers datasets tqdm numpy nltk pillow
RUN cd /workspace 

RUN apt update
RUN apt install vim -y

CMD ["bash", "-c", "wandb login && exec bash"]