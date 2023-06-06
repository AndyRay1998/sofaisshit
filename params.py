#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

class Simu:
    gravity=[0,0,0] # mm/s2
    dt= 0.01 #0.01
    SETUP_COMMUNICATION = 0 # connect to the real robot
    SETUP_HAPTICS = 1 # connect to the Touch X
    CableRatioKeypress = 0.1
    RobMoveRatioKeypress = 1

class Robot:
    ## Mecanical
    totalLengthRigTube = 100
    totalLengthOuter = 18
    totalLengthInner = 58
    totalLengthTool = 78
    initialLength = '18 21 0' # The length of the beam can not be reduced if it is shorter than initialLength because of the mechanical limitation
    # initialLengthTool = '24'
    nbPointCableModelInner = 50
    
    nbPointCableModelOuter = 50
    radiusRigTube = 3.5
    radiusOuter = 2.8
    radiusInner = 1.7
    radiusTool = 1.0
    pose="0 0 1  0  0  0  0"
    poseTool="0 0 0  0  0  0  0"
    poseRigTube="-100 0 0  0  0  0  0"

    numEdgesCollisRigTube = 20 # for rigid tube
    numEdgesCollisGuide = 9 # for outer catheter (Guide)
    numEdgesCollisCath = 29 # for inner catheter (Cath)
    numEdgesCollisTool = 39 # for flexible tool (tool)
 

class Brain:
    SPARSEGRID = 0
    rotation= '90. -55. -15' # "0 -43 20" # '90. -55. -15' FOR SPARSE GRID
    translation= '120 -110 -125' # '200 -130 -130' #'120 -110 -125 ' FOR SPARSE GRID

class Skull:
    rotation= '50 0 -110'
    translation= '315  105  -275'
