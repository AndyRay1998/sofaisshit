#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Sofa.Core
import Sofa.constants.Key as Key
import numpy as np
import time
import socket
import copy
import params
import math


class ControllerDisplacement(Sofa.Core.Controller):

    def __init__(self, *a, **kw):
        Sofa.Core.Controller.__init__(self, *a, **kw)
        self.name='ControllerDisplacement'
        self.node = kw["node"]
        self.displacement1 = 0
        self.displacement2 = 0
        self.displacement3 = 0
        self.displacement4 = 0
        self.displacement5 = 0
        self.displacement6 = 0

        self.xtipOIni = self.node.InstrumentCombined.m_ircontroller.xtip.value[0] # outer catheter
        self.xtipIIni = self.node.InstrumentCombined.m_ircontroller.xtip.value[1] # inner catheter
        # 此处的值=绳子总长-初始杆长. 即为pullPoint的x坐标值
        # inner
        self.node.InstrumentCombined.Cable1.cableConstraint1.value.value = [params.Robot.totalLengthInner-self.xtipIIni]
        self.node.InstrumentCombined.Cable2.cableConstraint2.value.value = [params.Robot.totalLengthInner-self.xtipIIni]
        self.node.InstrumentCombined.Cable3.cableConstraint3.value.value = [params.Robot.totalLengthInner-self.xtipIIni]
        # outer
        self.node.InstrumentCombined.Cable4.cableConstraint4.value.value = [params.Robot.totalLengthOuter-self.xtipOIni]
        self.node.InstrumentCombined.Cable5.cableConstraint5.value.value = [params.Robot.totalLengthOuter-self.xtipOIni]
        self.node.InstrumentCombined.Cable6.cableConstraint6.value.value = [params.Robot.totalLengthOuter-self.xtipOIni]
        
        return

    def onAnimateBeginEvent(self, event): # called at each begin of animation step
        control1 = self.node.control1.con1.position.value[0]
        control2 = self.node.control2.con2.position.value[0]

        inputvalue1 = self.node.InstrumentCombined.Cable1.cableConstraint1.value
        inputvalue2 = self.node.InstrumentCombined.Cable2.cableConstraint2.value
        inputvalue3 = self.node.InstrumentCombined.Cable3.cableConstraint3.value
        inputvalue4 = self.node.InstrumentCombined.Cable4.cableConstraint4.value
        inputvalue5 = self.node.InstrumentCombined.Cable5.cableConstraint5.value
        inputvalue6 = self.node.InstrumentCombined.Cable6.cableConstraint6.value

        self.displacement1 = inputvalue1.value[0]
        self.displacement2 = inputvalue2.value[0]
        self.displacement3 = inputvalue3.value[0]
        self.displacement4 = inputvalue4.value[0]
        self.displacement5 = inputvalue5.value[0]
        self.displacement6 = inputvalue6.value[0]
        

        ###### moving forward/backward
        self.displacement1 = inputvalue1.value[0] + control1[2]
        self.displacement2 = inputvalue2.value[0] + control1[2]
        self.displacement3 = inputvalue3.value[0] + control1[2]

        if control1[2] > 0:
            xtipI = self.node.InstrumentCombined.m_ircontroller.xtip.value[1] # inner catheter
            xtipILeft = params.Robot.totalLengthInner-xtipI
            if xtipILeft < 0 or xtipILeft == 0:
                self.displacement1 = inputvalue1.value[0]
                self.displacement2 = inputvalue2.value[0]
                self.displacement3 = inputvalue3.value[0]
                # vector as row. This is for xtip
                list1 = [self.xtipOIni, xtipI]
                list1[1] = params.Robot.totalLengthInner
                # Xtip have to cooradiate with the cable
                vector = np.array(list1)
                self.node.InstrumentCombined.m_ircontroller.xtip.value = vector
        if control1[2] < 0:
            xtipI = self.node.InstrumentCombined.m_ircontroller.xtip.value[1] # inner catheter
            xtipILeft = self.xtipIIni-xtipI
            if xtipILeft > 0 or xtipILeft == 0:
                self.displacement1 = inputvalue1.value[0]
                self.displacement2 = inputvalue2.value[0]
                self.displacement3 = inputvalue3.value[0]
                # vector as row. This is for xtip
                list1 = [self.xtipOIni, xtipI]
                list1[1] = self.xtipIIni
                # Xtip have to cooradiate with the cable
                vector = np.array(list1)
                self.node.InstrumentCombined.m_ircontroller.xtip.value = vector

        ###### bending
        self.displacement1 -= control1[1]/2
        self.displacement2 += control1[1]/2/math.sqrt(3)
        self.displacement3 += control1[1]/2/math.sqrt(3)
        self.displacement2 -= control1[0]/2
        self.displacement3 += control1[0]/2

        self.displacement4 -= control2[1]/2
        self.displacement5 += control2[1]/2/math.sqrt(3)
        self.displacement6 += control2[1]/2/math.sqrt(3)
        self.displacement5 -= control2[0]/2
        self.displacement6 += control2[0]/2


        xtipO = self.node.InstrumentCombined.m_ircontroller.xtip.value[0] # outer catheter
        xtipI = self.node.InstrumentCombined.m_ircontroller.xtip.value[1] # inner catheter
        # vector as row. This is for xtip
        list1 = [xtipO, xtipI]
        xtipI = xtipI - control1[2]
        list1[1] = xtipI
        
        inputvalue1.value = [self.displacement1]
        inputvalue2.value = [self.displacement2]
        inputvalue3.value = [self.displacement3]
        inputvalue4.value = [self.displacement4]
        inputvalue5.value = [self.displacement5]
        inputvalue6.value = [self.displacement6]
        
        # Xtip have to coordinate with the cable
        vector = np.array(list1)
        self.node.InstrumentCombined.m_ircontroller.xtip.value = vector

        return