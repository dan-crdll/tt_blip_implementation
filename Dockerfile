# Containerized environment to train the model
FROM pytorch/pytorch

WORKDIR /ws
COPY . /ws

RUN cd /ws
RUN pip install --no-cache-dir lightning datasets transformers wandb

CMD [ "python", "./train_bi_dec.py" ]


# HOW TO RUN

# To build the image:
# docker build -t <tag> .

# Be sure to have a file with the following line (for logging):
# WANDB_API_KEY=<wandb secret>

# To run:
# docker run -it --env-file <env_file_path> <tag> bash

# Train directly without entering bash
# docker run --env-file <env_file_path> <tag>

# To copy saved state dictionary
# docker cp <container_id>:/ws/model_state_dict.pth <host_directory>
# to get container_id run: docker ps