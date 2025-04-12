FROM python

RUN mkdir /workspace
WORKDIR /workspace 

RUN pip install torch lightning wandb transformers datasets tqdm numpy nltk pillow

RUN apt update
RUN apt install vim -y

RUN cd /workspace

CMD ["bash", "-c", "git clone https://github.com/dan-crdll/tt_blip_implementation.git && wandb login && exec bash"]