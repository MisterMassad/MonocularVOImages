**Requirements:**



The following packages must be installed using pip:

\- numpy

\- opencv-python

\- vispy

\- scipy

\- PyQt6



**Important: vispy instructions:**



Left Click + Drag - Rotation

Left Click + Shift + Drag - Move around

Right click + drag or Mouse wheel – Zoom in / out



**How to run the script:**



If "Dataset\_VO" is present in the directory:



python VO\_Est.py



Else:



python VO\_Est.py --data\_dir "path/to/Dataset\_VO"





**Outputs:**

\- traj\_estimation.txt              (raw trajectory in TUM format)

\- traj\_estimation\_smooth.txt       (smoothed translation only, rotations unchanged)



**To exit code:**



Esc on both windows

