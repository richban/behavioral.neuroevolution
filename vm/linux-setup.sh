#!/usr/bin/env bashi

# This script performs initiall setup for the
# project and install various dependencies.
# Script is inspired from https://github.com/davla/bash-util

#####################################################
#
#                   Privileges
#
#####################################################

# Checking for root privileges: if don't
# have them, recalling this script with sudo
if [[ $EUID -ne 0 ]]; then
  echo 'This script needs to be run as root'
  sudo bash "$0" "$@"
  exit 0
fi

#####################################################
#
#               Clean & upgrade
#
#####################################################

# Cleaning undesired packages,
# so that they won't be upgraded
clean

# Updating
apt-get update
apt-get upgrade

#####################################################
#
#           Installing packages
#
#####################################################

apt install python python3 python-pip python-gtk2 vim git
[[ $? -ne 0 ]] && exit 1

#####################################################
#
#                   Install  Aseba
#
#####################################################

sudo apt-get install qttools5-dev-tools \
                     qttools5-dev \
                     qtbase5-dev \
                     qt5-qmake \
                     libqt5help5 \
                     libqt5opengl5-dev \
                     libqt5svg5-dev \
                     libqt5x11extras5-dev \
                     libqwt-qt5-dev \
                     libudev-dev \
                     libxml2-dev \
                     libsdl2-dev \
                     libavahi-compat-libdnssd-dev \
                     python-dev \
                     libboost-python-dev \
                     doxygen \
                     cmake \
                     g++ \
                     git \
                     make \

cd mkdir ~/Developer && cd Developer
git clone --recursive https://github.com/aseba-community/aseba.git
cd aseba

# Building Aseba
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="<path of qt>;<path of bonjour>" ..
make

# User permissions
usermod -a -G dialout $USER
newgrp dialout
