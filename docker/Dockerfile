FROM nitincypher/docker-ubuntu-python-pip

RUN apt-get update && apt-get install -y lsb-release software-properties-common

RUN  add-apt-repository ppa:stephane.magnenat/`lsb_release -c -s` \
    && apt-get update \
    && apt-get install -y aseba

#RUN adduser $USER dialout

RUN asebamedulla &

