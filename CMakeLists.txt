cmake_minimum_required(VERSION 3.0.2)
project(mrmg_moveit_planning)

## This is a (mostly) Python catkin package. The planners and the MoveIt backend
## live in the `multi_robot_multi_goal_planning` Python package under `src/`, and
## are exposed to catkin via `catkin_python_setup()` (see setup.py). The ROS nodes
## are thin executables in `scripts/`.

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

#############
## Install ##
#############

## Python rosrun entry points
catkin_install_python(PROGRAMS
  scripts/mrmg_plan_node
  scripts/mrmg_replay_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Resources (launch files, MoveIt config, RViz, robot description, meshes,
## example problem specs)
foreach(dir launch config rviz urdf meshes)
  install(DIRECTORY ${dir}/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach()
