# RoMAN2021
A repository for reproducing the results presented in the ICDL-2023 submission.

> **Abstract:**  Humans can adapt to unexpected situations while interacting with others by performing dynamic decision-making. This skill is critical, especially in social interaction; for example, a student can adapt to the teaching style of the interaction partner, or a teacher can change the guiding strategy based on the student's learning progress. Despite the importance of this ability in humans, only a few studies target endowing artificial agents with dynamic decision-making skills. Moreover, only limited studies address this skill in the context of robot-robot interaction. In this study, we design robot-robot interaction experiments where a learner robot (Nao) interacts with a partner robot (Pepper) with adaptive and fixed guiding strategies to help the learner robot perform a sequential visual recalling task. The learner robot is equipped with four cognitive modules: auto-associative memory to recall visual patterns and derive the cost of visual recall (i.e., cognitive load), internal reward module to perform the task while minimizing cognitive load, self-other monitoring to track its individual versus partner assisted performance and dynamic decision-making to decide whether the partner robot support is beneficial. Here the learner robot aims to experience less cognitive load to complete the interactive task. Overall, the initial results show that dynamic decision-making based on self-other monitoring leads to a better average performance than interaction in fixed (i.e., non-dynamic) conditions.

## Folder and file descriptions
+ **gamegrid:** this folder contains the images to create the scene for visual processing, and
+ **gameimg:** contains the visual patterns to train associative memory, etc.   
+ **outputs:** this contains subfolders for the results in .npy format and visualizations in .pnf format.
+ **trainnao.py:** runs the pre-run training
+ **Naoexpgrids.py:** runs the experiments and contains the specifications for the experimental runs and safes all required outputs
+ **hopfieldnetwork.py:** contains the implementation of the auto-associative memory module
+ **SARSAlineteach.py** contains the implementation of the internal reward module for a reinforcement learning algorithm embedded with the self-other monitoring module and the dynamic-decision making module
+ **SARSAlineteach_nonadapt.py:** contains the implementation of the internal reward module for a reinforcement learning algorithm without adaptivity
