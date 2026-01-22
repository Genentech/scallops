FROM python:3.12-slim

ENV AWS_RETRY_MODE=adaptive \
    AWS_MAX_ATTEMPTS=10 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -qq --no-install-recommends -y \
      build-essential \
      git \
      ca-certificates

WORKDIR /app
COPY requirements.txt requirements.ufish.txt ./
RUN pip install -q --upgrade pip && pip install -q --no-cache-dir -r requirements.txt -r requirements.ufish.txt
COPY . ./
RUN pip install .
RUN apt-get remove -qq -y build-essential git && \
    apt-get autoremove -qq -y && apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
