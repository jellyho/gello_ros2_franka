#!/bin/bash
# setup_env.sh

BASHRC="$HOME/.bashrc"
MARKER="## GELLO ENVIRONMENT SETTINGS"

# Remove existing GELLO blocks if any
sed -i "/$MARKER/,/## END GELLO/d" "$BASHRC"

# Add fresh settings
echo "$MARKER" >> "$BASHRC"
echo "source /opt/ros/humble/setup.bash" >> "$BASHRC"
echo "export PYTHONPATH=/home/jellyho/workspace/miniconda/envs/gello/lib/python3.10/site-packages:\$PYTHONPATH" >> "$BASHRC"
echo "alias init='source ~/workspace/ros2_ws/install/setup.bash'" >> "$BASHRC"
echo "alias build='source ~/workspace/ros2_ws/install.sh'" >> "$BASHRC"
echo "alias gello='ros2 launch gello_driver gello.launch.py'" >> "$BASHRC"
echo "## END GELLO" >> "$BASHRC"

echo "Environment setup complete in .bashrc"
echo "Please run: source ~/.bashrc"
