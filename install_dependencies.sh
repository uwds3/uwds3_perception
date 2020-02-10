pip install -r requirements.txt --user
sudo apt install ros-${ROS_DISTRO}-joint-state-publisher-gui
echo "#To override ROS opencv-python package with newer version:" >> ~/.bashrc
echo "export PYTHONPATH='\${HOME}/.local/lib/python2.7/site-packages'" >> ~/.bashrc
