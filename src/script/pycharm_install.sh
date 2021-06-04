#!/bin/bash
# install pycharm tar file
wget "https://download.jetbrains.com/python/pycharm-community-2020.3.1.tar.gz?_ga=2.208079378.1482907034.1608610329-1913990257.1603182848" -O /home/appuser/pycharm.tar.gz

# decompress pycharm
tar -xvf /home/appuser/pycharm.tar.gz -C /home/appuser
rm -rf /home/appuser/pycharm.tar.gz

# alias pycahrm
echo "alias pycharm='bash /home/appuser/pycharm-community-2020.3.1/bin/pycharm.sh'" >> ~/.bashrc
source ~/.bashrc