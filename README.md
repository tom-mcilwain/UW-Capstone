# Capstone Project - University of Washington

Title: Designing a High-Dimensional Environment for Evaluating Decoder Adaptation Algorithm Performance in Neural Interfaces

Abstract: Brain-machine interfaces (BMIs) are being heavily researched at the moment for their incredibly ambitious applications such as controlling a robotic arm [1]. The effectiveness of a BMI can be heavily influenced by the method of decoding, the way in which neural activity is processed and mapped to a response on a computer. A poor decoder will lead to poor performance using the neural interface. To combat this, recently there have been a lot of studies surrounding the use of machine-learning to improve the decoders over time [2, 3, 4]. Closed-loop decoder adaptation (CLDA) is used to adapt the decoder in a closedloop neural interface setting in which the user is sending a signal and receiving feedback, and CLDA algorithms have been heavily researched in low-dimensionality settings. However, many BMI applications involve the user controlling multiple dimensions, thus these algorithms need to be tested when the user is performing a more complex task. This study outlines building an environment in which to test three CLDA algorithms (Batch algorithm, SmoothBatch algorithm, and Adaptive Kalman filter) in higher dimensions. The higher dimensional task is an expansion of a normal center-out task. A traditional center-out task involves the user controlling a cursor in two dimensions moving to and from a target on the screen. This has been expanded to include two cursors to double the amount of dimensions that the user is controlling. The cognitive difficulty of paying attention to two cursors is quantified using a two-cursor, one-dimension task in which two cursors both move in one dimension. Instead of a BMI, the study uses a kinematic interface, where a motion-capture Cyberglove records hand movements which are used to represent neural activity to control the cursor. The result of this study is an environment that is capable of testing decoder adaptation algorithms using a kinematic interface.
