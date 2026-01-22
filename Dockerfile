FROM python:3.12-slim

ENV AWS_RETRY_MODE=adaptive \
    AWS_MAX_ATTEMPTS=10 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
      build-essential \
      git \
      ca-certificates

RUN pip install -q --upgrade pip && pip install -q --no-cache-dir -r ./requirements.txt -r ./requirements.ufish.txt
RUN pip install .
RUN apt-get remove -y build-essential git && \
    apt-get autoremove -y &&  apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
