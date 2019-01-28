#!/usr/bin/env bash

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

# Updating
apt-get update
apt-get upgrade

#####################################################
#
#           Installing packages
#
#####################################################

apt install python python3 python-pip python-gtk2 vim git \
  make build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
  xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev curl python-dbus
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
                     autoconf \
                     libreadline-dev \
                     libncurses-dev \
                     libssl-dev \
                     libyaml-dev \
                     libxslt-dev \
                     libffi-dev \
                     libtool \
                     unixodbc-dev
[[ $? -ne 0 ]] && exit 1

cd ~/ && mkdir ~/Developer && cd Developer
git clone --recursive https://github.com/aseba-community/aseba.git
cd aseba

# Building Aseba
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="<path of qt>;<path of bonjour>" ..
make

# User permissions
usermod -a -G dialout $USER
newgrp dialout

#################################
# install asdf
#################################

if [[ ! -d $HOME/.asdf ]]; then
  git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.6.3
  echo -e '\n. $HOME/.asdf/asdf.sh' >> ~/.bashrc
  echo -e '\n. $HOME/.asdf/completions/asdf.bash' >> ~/.basrc
fi

if [[ $? != 0 ]]; then
exit 2
fi

#####################################################
#
#                   Setup Python
#
#####################################################

if [[ ! -d $HOME/.asdf/plugins/python ]]; then
  $HOME/.asdf/bin/asdf plugin-add python https://github.com/danhper/asdf-python.git
fi

if [[ $? != 0 ]]; then
  exit 2
fi

if [[ ! -d $HOME/.asdf/installs/python ]]; then
  $HOME/.asdf/bin/asdf install python 3.7.2
fi

if [[ $? != 0 ]]; then
  exit 2
fi

$HOME/.asdf/bin/asdf global python 3.7.2
if [[ $? != 0 ]]; then
  exit 2
fi

