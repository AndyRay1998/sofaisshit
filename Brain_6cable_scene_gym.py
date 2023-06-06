import Sofa
import time
import SofaRuntime
import Sofa.Gui

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import copy
import params

import sys
sys.path.append('./controller')
from brain_6cable_gym_control import ControllerDisplacement

translation3='-0.8  -5  3.5'
rotation3='0 90 0'

def createLinePoints(x_max, numLines):
	step = x_max/numLines
	pos_out = [0]*3*(numLines+1)
	x_c=0

	for i in range(numLines):
		x_c=x_c+step
		pos_out[3*i+3]= x_c

	return pos_out

def transformTableInString(Table):
	sizeT =  len(Table)
	strOut= ' '

	for p in range(sizeT):
		strOut = strOut+ str(Table[p])+' '

	return strOut


class Brain_6cable_scene(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self):
        self.root = Sofa.Core.Node('root')
        self.createScene(self.root)
        Sofa.Simulation.init(self.root)

        for _ in range(100):
            Sofa.Simulation.animate(self.root, self.root.getDt())
        
        # TODO: determine action amplitude
        bend_amp = 0.1
        insert_amp = 0.2
        self.action_space = spaces.Box(
            low=np.array([-bend_amp, -bend_amp, -insert_amp,    -bend_amp, -bend_amp]), 
            high=np.array([bend_amp, bend_amp, insert_amp,    bend_amp, bend_amp]), shape=(5,), dtype=np.float32
        )

        high = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.step_count = 0
        self.termination = False
        self.seed(0)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #### manually clipping actions
        # the range value coincides with self.action_space
        bend_amp = 0.1
        insert_amp = 0.2
        action = np.clip(action, np.array([-bend_amp, -bend_amp, -insert_amp,    -bend_amp, -bend_amp]), 
                         np.array([bend_amp, bend_amp, insert_amp,    bend_amp, bend_amp]))

        #### modify artificial mechanical object to convey control signal
        with self.root.control1.con1.position.writeableArray() as con1:
                con1[0][0] = action[0] # x bending 
                con1[0][1] = action[1] # y bending
                con1[0][2] = action[2] # insertion
        with self.root.control2.con2.position.writeableArray() as con2:
                con2[0][0] = action[3] # x bending 
                con2[0][1] = action[4] # y bending 
                # con2[0][2] = action[5]

        # 1 step forward
        Sofa.Simulation.animate(self.root, self.root.getDt())

        ############## termination ####################
        # in case of chattering, i.e. the velocity norm is too large
        # TODO
        if self.step_count > 1000: 
              self.termination = True
              self.step_count = 0
        else:
              self.termination = False
        
        ############## reward function ####################
        ## TODO: reward shaping
        brain_force_norm = np.linalg.norm(self.root.brain.dofs.force.value)
        position_error = np.linalg.norm(self.root.target.target_pos.position.value[0][:3] - self.root.InstrumentCombined.DOFs.position.value[-1][:3])
        control_penalty = np.linalg.norm(action)
        reward = - 0.*brain_force_norm - 10*position_error - 0.*control_penalty
        
        # self.state = self.root.brain.dofs.position.value
        # self.root.brain.dofs.velocity.value
        # self.root.brain.dofs.force.value

        # print(self.root.InstrumentCombined.DOFs.position.value[-1][:3], position_error, self.root.InstrumentCombined.m_ircontroller.xtip.value[1])
        # print(np.linalg.norm(self.root.InstrumentCombined.DOFs.velocity.value))
        # print(brain_force_norm)

        self.step_count += 1

        return self._get_obs(), reward, self.termination, {}

    def reset(self):
        # delete and then recreate a new root node
        del self.root
        self.root = Sofa.Core.Node('root')
        self.createScene(self.root)
        Sofa.Simulation.init(self.root)
        return self._get_obs()

    def _get_obs(self):
        # the state should satisfy Markov assumption
        # 6 cable lengths + 3 tip position
        cable1 = np.array([self.root.InstrumentCombined.Cable1.cableConstraint1.value[0]])
        cable2 = np.array([self.root.InstrumentCombined.Cable2.cableConstraint2.value[0]])
        cable3 = np.array([self.root.InstrumentCombined.Cable3.cableConstraint3.value[0]])
        cable4 = np.array([self.root.InstrumentCombined.Cable4.cableConstraint4.value[0]])
        cable5 = np.array([self.root.InstrumentCombined.Cable5.cableConstraint5.value[0]])
        cable6 = np.array([self.root.InstrumentCombined.Cable6.cableConstraint6.value[0]])
        tip = self.root.InstrumentCombined.DOFs.position.value[-1][:3]
        tmp = np.concatenate((cable1, cable2), axis=0)
        tmp = np.concatenate((tmp, cable3), axis=0)
        tmp = np.concatenate((tmp, cable4), axis=0)
        tmp = np.concatenate((tmp, cable5), axis=0)
        tmp = np.concatenate((tmp, cable6), axis=0)
        self.state = np.concatenate((tmp, tip), axis=0)
        
        return self.state

    def createScene(self, rootNode):
        rootNode.addObject('RequiredPlugin',name='BeamAdapter', pluginName='BeamAdapter ')
        rootNode.addObject('RequiredPlugin',name='SofaPython3', pluginName='SofaPython3')
        rootNode.addObject('RequiredPlugin',name ='SoftRobots', pluginName='SoftRobots')
        # rootNode.addObject('RequiredPlugin',name ='GPUSComputing', pluginName='SofaCUDA')
        # rootNode.addObject('AnimationLoopParallelScheduler', name="mainLoop", threadNumber="10")
        # rootNode.addObject('RequiredPlugin',name ='MultiThreading computing', pluginName='MultiThreading')
	 
        rootNode.addObject('BackgroundSetting', color='0 0.168627 0.211765')
        rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideMappings hideForceFields showInteractionForceFields hideWireframe')
        rootNode.addObject('RequiredPlugin', pluginName=["Sofa.Component.AnimationLoop",
	                                                 # Needed to use components FreeMotionAnimationLoop
                                                     "Sofa.Component.Collision.Detection.Algorithm",
                                                     # Needed to use components BVHNarrowPhase, BruteForceBroadPhase, DefaultPipeline
                                                     "Sofa.Component.Collision.Detection.Intersection",
                                                     # Needed to use components LocalMinDistance
                                                     "Sofa.Component.Collision.Geometry",
                                                     # Needed to use components LineCollisionModel, PointCollisionModel, TriangleCollisionModel
                                                     "Sofa.Component.Collision.Response.Contact",
                                                     # Needed to use components DefaultContactManager
                                                     "Sofa.Component.Constraint.Lagrangian.Correction",
                                                     # Needed to use components GenericConstraintCorrection, UncoupledConstraintCorrection
                                                     "Sofa.Component.Constraint.Lagrangian.Solver",
                                                     # Needed to use components GenericConstraintSolver
                                                     "Sofa.Component.IO.Mesh",
                                                     # Needed to use components MeshOBJLoader, MeshSTLLoader, MeshVTKLoader
                                                     "Sofa.Component.LinearSolver.Direct",
                                                     # Needed to use components SparseLDLSolver
                                                     "Sofa.Component.LinearSolver.Iterative",
                                                     # Needed to use components CGLinearSolver
                                                     "Sofa.Component.Mass",  # Needed to use components UniformMass
                                                     "Sofa.Component.ODESolver.Backward",
                                                     # Needed to use components EulerImplicitSolver
                                                     "Sofa.Component.SolidMechanics.FEM.Elastic",
                                                     # Needed to use components TetrahedronFEMForceField
                                                     "Sofa.Component.Topology.Container.Constant",
                                                     # Needed to use components MeshTopology
                                                     "Sofa.Component.Topology.Container.Dynamic",
                                                     # Needed to use components TetrahedronSetTopologyContainer, TetrahedronSetTopologyModifier
                                                     "Sofa.Component.Visual",  # Needed to use VisualStyle
                                                     "Sofa.GL.Component.Rendering3D", # Needed to use OglModel
                                                     "Sofa.GUI.Component", # Needed to use AttachBodyButtonSetting
                                                     "Sofa.Component.Constraint.Projective", # Needed to use components [FixedConstraint]
                                                     "Sofa.Component.Mapping.Linear", # Needed to use components [IdentityMapping]
                                                     "Sofa.Component.Setting", # Needed to use components [BackgroundSetting]
                                                     "Sofa.Component.SolidMechanics.Spring", # Needed to use components [RestShapeSpringsForceField]
                                                     "Sofa.Component.StateContainer", # Needed to use components [MechanicalObject]
                                                     "Sofa.Component.Topology.Container.Grid", # Needed to use components [RegularGridTopology]
                                                     "Sofa.Component.Topology.Mapping", # Needed to use components [Edge2QuadTopologicalMapping]
                                                     'Sofa.Component.Engine.Transform',
                                                     'Sofa.Component.Engine.Select',
                                                     ])
        rootNode.findData('dt').value=0.01
        rootNode.findData('gravity').value=params.Simu.gravity

        rootNode.addObject('DefaultVisualManagerLoop')
        rootNode.addObject('FreeMotionAnimationLoop', parallelCollisionDetectionAndFreeMotion='true',parallelODESolving='true')
        rootNode.addObject('GenericConstraintSolver', name='GCS', tolerance="1e-6", maxIterations="500", resolutionMethod="UnbuildGaussSeidel", computeConstraintForces='true')
        rootNode.addObject('DefaultPipeline', depth="15", verbose="0", draw="1")
        rootNode.addObject('BruteForceBroadPhase', name='N2')
        rootNode.addObject('BVHNarrowPhase')
        rootNode.addObject('DefaultContactManager', response="FrictionContactConstraint", responseParams="mu=0.65")
        rootNode.addObject('LocalMinDistance', name="Proximity", alarmDistance="0.5", contactDistance="0.3", angleCone="0.01")


	########################################
	# INTERVENTIONAL RADIOLOGY INSTRUMENTS (catheter)
	########################################
	########## This is actually a collision model, robotic environment ###########
        brain = rootNode.addChild('brain')
        
        # cochleaNode.addObject('MeshVTKLoader', name='loader', filename='mesh/brain.VTK', flipNormals="false")
        brain.addObject('SparseGridTopology', name='loader',n=[10,10,10], fileTopology='mesh/brain.obj')
        brain.addObject("TransformEngine", name="translationEngine", template="Vec3d", rotation=params.Brain.rotation, translation=params.Brain.translation, input_position="@loader.position")
        brain.addObject("MechanicalObject", name="dofs", template="Vec3d", position="@translationEngine.output_position")
        brain.addObject('MeshTopology',src = '@loader')
        brain.addObject('EulerImplicitSolver',rayleighMass=0.1, rayleighStiffness=0.1)
        brain.addObject('CGLinearSolver',iterations="100", tolerance="1e-5" ,threshold="1e-5")
        
        # brain.addObject('HexahedronFEMForceField', name="FEM", youngModulus="0.1", poissonRatio="0.4", method="large",
        #                 updateStiffnessMatrix="false", printLog="0")
        brain.addObject('TetrahedronFEMForceField', youngModulus=1, poissonRatio=0.45, method='polar', computeVonMisesStress='1', showVonMisesStressPerNode='true', listening='1', updateStiffness='true')
        brain.addObject('UncoupledConstraintCorrection', useOdeSolverIntegrationFactors="0")
        brain.addObject('BoxROI', name='BoxROI', box=[[140,65,-95],[160,50,-110]], drawBoxes=False)
        brain.addObject('RestShapeSpringsForceField', points='@BoxROI.indices', stiffness=10)
        brain.addObject('UniformMass', totalMass=10)

        collisionmodel = brain.addChild('CollisionModel')
        collisionmodel.addObject('MeshOBJLoader', name='loader', filename='mesh/brain.obj', flipNormals="false") #surface points
        collisionmodel.addObject('TriangleSetTopologyContainer', src='@loader', name='container')
        collisionmodel.addObject('MechanicalObject', template='Vec3d', name='EnvCollisionMO')
        collisionmodel.addObject('TriangleCollisionModel')
        # collisionmodel.addObject('SphereCollisionModel',radius=1.5)
        collisionmodel.addObject('LineCollisionModel')
        collisionmodel.addObject('PointCollisionModel')
        collisionmodel.addObject('BarycentricMapping',input="@..",output="@EnvCollisionMO")

        visuBrainNode = brain.addChild('visuBrainNode')
        visuBrainNode.addObject('MeshOBJLoader',name='loader',filename='mesh/brain.obj')
        visuBrainNode.addObject('OglModel', name="Visual", src="@loader", color="0.75 0.75 0.75 0.4") #, color="1 1 1 0.15"
        visuBrainNode.addObject('BarycentricMapping',input="@..",output="@Visual")
	

        #############################flexible beam########################
        # inner
        topoLines_cath = rootNode.addChild('topoLines_cath') 
        topoLines_cath.addObject('WireRestShape', template="Rigid3d", printLog=False, name="CatheterRestShape", length=params.Robot.totalLengthInner,
										straightLength=params.Robot.totalLengthInner, densityOfBeams="50",
										numEdges=params.Robot.numEdgesCollisCath, numEdgesCollis= params.Robot.numEdgesCollisCath, youngModulus="1e3", youngModulusExtremity="1e2", radius = params.Robot.radiusInner)
	#"numEdgesCollis:" "number of Edges for the collision model"
        topoLines_cath.addObject('EdgeSetTopologyContainer', name="meshLinesCath")
        topoLines_cath.addObject('EdgeSetTopologyModifier', name="Modifier")
        topoLines_cath.addObject('EdgeSetGeometryAlgorithms', name="GeomAlgo", template="Rigid3d")
        topoLines_cath.addObject('MechanicalObject', template="Rigid3d", name="MOInner")
        
        # outer
        topoLines_guide = rootNode.addChild('topoLines_guide')
        topoLines_guide.addObject('WireRestShape', template='Rigid3d', printLog=False, name='GuideRestShape', length=params.Robot.totalLengthOuter,
										straightLength=params.Robot.totalLengthOuter, densityOfBeams='50',
										numEdges= params.Robot.numEdgesCollisGuide, numEdgesCollis=params.Robot.numEdgesCollisGuide,  youngModulus='1e3', youngModulusExtremity='1e2', radius = params.Robot.radiusOuter)
        topoLines_guide.addObject('EdgeSetTopologyContainer', name='meshLinesGuide')
        topoLines_guide.addObject('EdgeSetTopologyModifier', name='Modifier')
        topoLines_guide.addObject('EdgeSetGeometryAlgorithms', name='GeomAlgo',   template='Rigid3d')
        topoLines_guide.addObject('MechanicalObject', template='Rigid3d', name='MOOuter')
	
        # Define starting position
        RefStartingPos = rootNode.addChild('RefStartingPos')
        RefStartingPos.addObject('MechanicalObject', name="ReferencePos", template="Rigid3d", position="-100 0 0  0 0 0 0") # position="0 0 0  0  0  0  0"


        InstrumentCombined = rootNode.addChild('InstrumentCombined')
        InstrumentCombined.addObject('EulerImplicitSolver', rayleighStiffness="0.01", rayleighMass="0.03", printLog=False)
        InstrumentCombined.addObject('SparseLDLSolver', name='ldlsolveur', template='CompressedRowSparseMatrixMat3x3d')
        InstrumentCombined.addObject('RegularGridTopology', name="meshLinesCombined", nx="200", ny="1", nz="1")

        InstrumentCombined.addObject('MechanicalObject', template="Rigid3d", name="DOFs")
        InstrumentCombined.addObject('InterventionalRadiologyController', template="Rigid3d", name="m_ircontroller", printLog=False, xtip= params.Robot.initialLength, speed ='0', step="0.1", rotationInstrument="0", controlledInstrument="0", startingPos="@../RefStartingPos/ReferencePos.position", instruments="InterpolGuide InterpolCatheter")
        InstrumentCombined.addObject('WireBeamInterpolation', name="InterpolCatheter", WireRestShape="@../topoLines_cath/CatheterRestShape", radius="100", printLog=False)
        InstrumentCombined.addObject('AdaptiveBeamForceFieldAndMass', name="CatheterForceField", massDensity="0.000005", interpolation="@InterpolCatheter", printLog=False)
        InstrumentCombined.addObject('WireBeamInterpolation', name='InterpolGuide', WireRestShape='@../topoLines_guide/GuideRestShape', radius='200', printLog='0')
        InstrumentCombined.addObject('AdaptiveBeamForceFieldAndMass', name='GuideForceField', massDensity="0.000005", interpolation='@InterpolGuide', printLog=False)

        InstrumentCombined.addObject('LinearSolverConstraintCorrection', printLog=False, wire_optimization="true")
        InstrumentCombined.addObject("FixedConstraint", name='FixedConstraint')
        InstrumentCombined.addObject('RestShapeSpringsForceField', name="MeasurementFF", points="@m_ircontroller.indexFirstNode",  stiffness="1e15", recompute_indices="1", angularStiffness="1e15", external_rest_shape="@../RefStartingPos/ReferencePos", external_points="0", drawSpring="1", springColor="1 0 0 1")
        #########################################
        # Cables
        #########################################
        pos = createLinePoints(params.Robot.totalLengthInner,params.Robot.nbPointCableModelInner)
        indices = range(0,int(params.Robot.nbPointCableModelInner+1))
        pos_str=transformTableInString(pos)
        indices_str = transformTableInString(indices)
        cable1 = InstrumentCombined.addChild('Cable1') # inner
        cable1.addObject('VisualStyle', displayFlags="showInteractionForceFields")
        cable1.addObject('MechanicalObject' , template='Vec3d',name="cable1", position=pos_str, translation='0 0 -1.2')
        cable1.addObject('CableConstraint', name="cableConstraint1", indices=indices_str, pullPoint='0 0 -1.2', hasPullPoint='0', drawPullPoint = True, drawPoints = True)
        cable1.addObject("FixedConstraint", indices="0")
        cable1.addObject('AdaptiveBeamMapping', name='mapping', mapForces='false', mapMasses='false',useCurvAbs="1", printLog="0", interpolation="@../InterpolCatheter", input="@../DOFs",output="@cable1", isMechanical="true")

        cable2 = InstrumentCombined.addChild('Cable2') # inner
        cable2.addObject('VisualStyle', displayFlags="showInteractionForceFields")
        cable2.addObject('MechanicalObject' , template='Vec3d',name="cable2", position=pos_str, translation='0 -1.14 0.6')
        cable2.addObject('CableConstraint', name="cableConstraint2", indices=indices_str, pullPoint=' 0 -1.14 0.6', hasPullPoint='0', drawPullPoint = True, drawPoints =True)
        cable2.addObject('AdaptiveBeamMapping', name='mapping', mapForces='false', mapMasses='false',useCurvAbs="1", printLog="0", interpolation="@../InterpolCatheter", input="@../DOFs",output="@cable2", isMechanical="true")

        cable3 = InstrumentCombined.addChild('Cable3') # inner
        cable3.addObject('VisualStyle', displayFlags="showInteractionForceFields")
        cable3.addObject('MechanicalObject' , template='Vec3d',name="cable3", position=pos_str, translation='0 1.14 0.6')
        cable3.addObject('CableConstraint', name="cableConstraint3", indices=indices_str, pullPoint=' 0 1.14 0.6', hasPullPoint='0', drawPullPoint = True, drawPoints = True)
        cable3.addObject('AdaptiveBeamMapping', name='mapping', mapForces='false', mapMasses='false',useCurvAbs="1", printLog="0", interpolation="@../InterpolCatheter", input="@../DOFs",output="@cable3", isMechanical="true")

        pos2 = createLinePoints(params.Robot.totalLengthOuter,params.Robot.nbPointCableModelOuter)
        indices2 = range(0,int(params.Robot.nbPointCableModelOuter+1))
        pos_str2=transformTableInString(pos2)
        indices_str2 = transformTableInString(indices2)

        cable4 = InstrumentCombined.addChild('Cable4') # outer
        cable4.addObject('VisualStyle', displayFlags="showInteractionForceFields")
        cable4.addObject('MechanicalObject' , template='Vec3d',name="cable4", position=pos_str2, translation='0 0 -2.4')
        cable4.addObject('CableConstraint', name="cableConstraint4", indices=indices_str2, pullPoint='0 0 -2.4', hasPullPoint='0', drawPullPoint =True, drawPoints =True)
        cable4.addObject("FixedConstraint", indices="0")
        cable4.addObject('AdaptiveBeamMapping', name='mapping', mapForces='false', mapMasses='false',useCurvAbs="1", printLog="0", interpolation="@../InterpolGuide", input="@../DOFs",output="@cable4", isMechanical="true")

        cable5 = InstrumentCombined.addChild('Cable5') # outer
        cable5.addObject('VisualStyle', displayFlags="showInteractionForceFields")
        cable5.addObject('MechanicalObject' , template='Vec3d',name="cable5", position=pos_str2, translation='0 -2.08 1.2')
        cable5.addObject('CableConstraint', name="cableConstraint5", indices=indices_str2, pullPoint='0 -2.08 1.2', hasPullPoint='0', drawPullPoint = True, drawPoints =True) # drawPullPoint = False, drawPoints = False
        cable5.addObject("FixedConstraint", indices="0")
        cable5.addObject('AdaptiveBeamMapping', name='mapping', mapForces='false', mapMasses='false',useCurvAbs="1", printLog="0", interpolation="@../InterpolGuide", input="@../DOFs",output="@cable5", isMechanical="true")

        cable6 = InstrumentCombined.addChild('Cable6') # outer
        cable6.addObject('VisualStyle', displayFlags="showInteractionForceFields")
        cable6.addObject('MechanicalObject', template='Vec3d',name="cable6", position=pos_str2, translation='0 2.08 1.2')
        cable6.addObject('CableConstraint', name="cableConstraint6", indices=indices_str2, pullPoint='0 2.08 1.2', hasPullPoint='0', drawPullPoint = True, drawPoints = True)
        cable6.addObject("FixedConstraint", indices="0")
        cable6.addObject('AdaptiveBeamMapping', name='mapping', mapForces='false', mapMasses='false',useCurvAbs="1", printLog="0", interpolation="@../InterpolGuide", input="@../DOFs",output="@cable6", isMechanical="true")

        #########################################
        # InstrumentCombined custom controller
        #########################################
        InstrumentCombined.addObject(ControllerDisplacement(node=rootNode))

        #########################################
        # InstrumentCombined Collision model
        #########################################
        CollisInstrumentCombined = InstrumentCombined.addChild('CollisInstrumentCombined')
        CollisInstrumentCombined.addObject('EdgeSetTopologyContainer', name="collisEdgeSet")
        CollisInstrumentCombined.addObject('EdgeSetTopologyModifier', name="colliseEdgeModifier")
        CollisInstrumentCombined.addObject('MechanicalObject', name="CollisInstrumentCombinedMO")
        CollisInstrumentCombined.addObject('MultiAdaptiveBeamMapping', name="multimapp", ircontroller="../m_ircontroller", useCurvAbs="1", printLog="false")
        CollisInstrumentCombined.addObject('LineCollisionModel')
        CollisInstrumentCombined.addObject('PointCollisionModel')

        # visualization of outer
        VisuGuide = InstrumentCombined.addChild('VisuGuide')
        VisuGuide.activated = True
        VisuGuide.addObject('MechanicalObject', name='VisuGuideQuads')
        VisuGuide.addObject('QuadSetTopologyContainer', name='ContainerGuide')
        VisuGuide.addObject('QuadSetTopologyModifier', name='VisuGuideModifier' )
        VisuGuide.addObject('QuadSetGeometryAlgorithms', name='VisuGuideGeomAlgo', template='Vec3d')
        VisuGuide.addObject('Edge2QuadTopologicalMapping', nbPointsOnEachCircle='20', radius=params.Robot.radiusOuter, input='@../../topoLines_guide/meshLinesGuide', output='@ContainerGuide', flipNormals='true', printLog=False)
        VisuGuide.addObject('AdaptiveBeamMapping', name='visuMapGuide', useCurvAbs='1', printLog='0', interpolation='@../InterpolGuide', input='@../DOFs', output='@VisuGuideQuads', isMechanical='false')

        VisuGuideVisuOgl = VisuGuide.addChild('VisuGuideVisuOgl')
        VisuGuideVisuOgl.addObject('OglModel', name='VisuGuideVisual', color='red')
        VisuGuideVisuOgl.addObject('IdentityMapping', input='@../VisuGuideQuads', output='@VisuGuideVisual')
		
        # visualization of inner
        VisuCath = InstrumentCombined.addChild('VisuCath')
        VisuCath.activated = True
        VisuCath.addObject('MechanicalObject', name="Quads")
        VisuCath.addObject('QuadSetTopologyContainer', name="ContainerCath")
        VisuCath.addObject('QuadSetTopologyModifier', name="Modifier" )
        VisuCath.addObject('QuadSetGeometryAlgorithms', name="GeomAlgo", template="Vec3d")
        VisuCath.addObject('Edge2QuadTopologicalMapping', nbPointsOnEachCircle="20", radius=params.Robot.radiusInner, input="@../../topoLines_cath/meshLinesCath", output="@ContainerCath", flipNormals="true", printLog=False)
        VisuCath.addObject('AdaptiveBeamMapping', name="VisuMapCath", useCurvAbs="1", printLog=False, isMechanical="false",  interpolation="@../InterpolCatheter")
 
        VisuCathVisuOgl = VisuCath.addChild('VisuCathVisuOgl')
        VisuCathVisuOgl.addObject('OglModel',name="VisualCathOGL", src="@../ContainerCath", color='green')
        VisuCathVisuOgl.addObject('IdentityMapping', input="@../Quads", output="@VisualCathOGL")


        # use an artificial mechanical object to store per-step control signal
        control1 = rootNode.addChild("control1")
        control1.addObject('MechanicalObject', name="con1", template="Vec3d", position="0 0 0")
        control2 = rootNode.addChild("control2")
        control2.addObject('MechanicalObject', name="con2", template="Vec3d", position="0 0 0")

        # use an mechanical object to store the target XYZ position
        target = rootNode.addChild("target")
        target.addObject('MechanicalObject', name="target_pos", template="Vec3d", position="-0 5.65 0.")

        return 



if __name__ == '__main__':
    
    env = Brain_6cable_scene()
    observation = env.reset()

    for i in range(100):
        action = env.action_space.sample()  # this is where you would insert your policy
        action = np.array([0,0,0.1,0,0])
        observation, reward, terminated, info = env.step(action)
        # print(reward)
        # print(env.root.brain.dofs.velocity.value.shape)
        # print(env.root.InstrumentCombined.DOFs.velocity.value[-1])
        print(i, reward, observation[-3])
    print(observation)


# if __name__ == '__main__':
    
#     env1 = Brain_6cable_scene()
#     observation1 = env1.reset()

#     env2 = Brain_6cable_scene()
#     observation2 = env2.reset()

#     for _ in range(10):
#         print(env1.step(np.array([0,0,1,0,0])))
#         print(env1.step(np.array([0,0,-1,0,0])))
        

#     for _ in range(0):
#         action_shape = env1.action_space.shape
#         epsilon=0.001
#         alpha = 0.01
#         # action = np.array([0,0,0.1,0,0])

#         # # randomized smoothing
#         # estim_grad = np.zeros(shape=action_shape)
#         # for _ in range(6):
#         #     radius = np.array([0.1,0.1,0.2,0.1,0.1]) * epsilon
#         #     base_action = np.zeros(shape=radius.shape)
#         #     rand_action = np.random.uniform(low=-radius, high=radius, size=(radius.shape))
#         #     _, cost_rand, _, _ = env2.step(rand_action)
#         #     _, cost_base, _, _ = env1.step(base_action)
#         #     estim_grad += (cost_rand-cost_base)*rand_action * base_action.shape[-1] / radius
#         #     # reset env2
#         #     _, _, _, _ = env2.step(-rand_action)
        
#         # finite difference
#         estim_grad = np.zeros(shape=action_shape)
#         # for i in range(action_shape[0]):
#         # base_action = np.zeros(shape=action_shape)
#         i = 2
#         rand_action = np.zeros(shape=action_shape)
#         rand_action[i] = epsilon
#         _, cost_rand1, _, _ = env2.step(0.5*rand_action)
#         _, cost_rand2, _, _ = env2.step(-rand_action)
#         # _, cost_base, _, _ = env1.step(base_action)
#         estim_grad[i] = (cost_rand1-cost_rand2) / epsilon
#         # reset env2
#         _, _, _, _ = env2.step(-0.5*rand_action)

#         action = -alpha * estim_grad
#         obs1, reward1, _, _ = env1.step(action)
#         obs2, reward2, _, _ = env2.step(action)
#         print(reward1)
#         # print(env.root.brain.dofs.velocity.value.shape)
#         # print(env.root.InstrumentCombined.DOFs.velocity.value[-1])
#     print(obs1)