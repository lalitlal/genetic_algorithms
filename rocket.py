from Box2D import (b2FixtureDef, b2PolygonShape, b2Filter)

from brain import Brain
from chromosome import (PControlChromosome, PDControlChromosome, PIDControlChromosome, SHACTChromosome, IslandChromosome)

from math import (atan2, floor, pi, radians)
from numpy import (array, cos, exp, random, sin)


class Rocket():

    NO_MASK_FILTER = 0x0001
    ROCKET_FILTER = 0x0002

    def __init__(self, world, chromosome):
        super(Rocket, self).__init__()

        self.world = world
        self.chromosome = chromosome

        if isinstance(self.chromosome, SHACTChromosome):
            self.updateFunction = self.updateSHACT
            self.brain = Brain(self.chromosome)
        elif isinstance(self.chromosome, PControlChromosome):
            self.updateFunction = self.updatePController
        elif isinstance(self.chromosome, PDControlChromosome):
            self.updateFunction = self.updatePDController
        elif isinstance(self.chromosome, PIDControlChromosome):
            self.updateFunction = self.updatePIDController
        elif isinstance(self.chromosome, IslandChromosome):
            self.updateFunction = self.updateIslandGA
            self.brain = Brain(self.chromosome)

        r_w = self.chromosome['width']
        r_h = self.chromosome['height']
        l_w, l_h = 0.3, 1.6
        r_o = r_h * 0.5 + 1.0

        filt = b2Filter(
            categoryBits=self.ROCKET_FILTER,
            maskBits=self.NO_MASK_FILTER)
        frame_fixture = b2FixtureDef(
            shape=b2PolygonShape(box=(r_w / 2, r_h / 2)),
            density=1,
            filter=filt)
        self.frame = self.world.CreateDynamicBody(
            position=(0, r_o),
            fixtures=frame_fixture)

        # Landing legs.
        left_angle = radians(self.chromosome['left.leg.angle'])
        right_angle = radians(self.chromosome['right.leg.angle'])

        left_origin = (
            -(r_w * 0.5) + sin(left_angle) * l_h * 0.5,
            r_o - r_h * 0.5 + 0.4 - cos(left_angle) * l_h * 0.5)
        right_origin = (
            -(r_w * -0.5) + sin(right_angle) * l_h * 0.5,
            r_o - r_h * 0.5 + 0.4 - cos(right_angle) * l_h * 0.5)

        self.leftLeg, self.leftNULL = self.createObject((0.15, 0.8),
            left_origin, left_angle)
        self.rightLeg, self.rightNULL = self.createObject((0.15, 0.8),
            right_origin, right_angle)

        # Bottom thruster.
        bottom_origin = (0, r_o - r_h * 0.5)
        self.bottomThruster, self.bottomThrusterJoint = self.createObject(
            (0.30, 0.30), bottom_origin, radians(
                self.chromosome['bottom.thrust.angle']))

        # Top thrusters.
        left_origin = (r_w * -0.5, r_o + r_h * 0.5 - 0.2)
        right_origin = (r_w * 0.5, r_o + r_h * 0.5 - 0.2)
        self.leftThruster, self.leftNULL = self.createObject((0.075, 0.15),
            left_origin, radians(self.chromosome['left.thrust.angle']))
        self.rightThruster, self.rightNULL = self.createObject((0.075, 0.15),
            right_origin, radians(self.chromosome['right.thrust.angle']))

    def teardown(self):
        self.world.DestroyBody(self.frame)
        self.world.DestroyBody(self.bottomThruster)
        self.world.DestroyBody(self.leftThruster)
        self.world.DestroyBody(self.rightThruster)
        self.world.DestroyBody(self.leftLeg)
        self.world.DestroyBody(self.rightLeg)

    def createObject(self, size, position, angle):
        filt = b2Filter(
            categoryBits=self.ROCKET_FILTER,
            maskBits=self.NO_MASK_FILTER)
        fixture = b2FixtureDef(
            shape=b2PolygonShape(box=size),
            density=1,
            filter=filt)
        obj = self.world.CreateDynamicBody(
            position=position,
            fixtures=fixture)
        jnt = self.world.CreateRevoluteJoint(
            bodyA=self.frame,
            bodyB=obj,
            anchor=position,
            collideConnected=False,
            lowerAngle=angle,
            upperAngle=angle,
            enableLimit=True,
            enableMotor=False)
        return obj, jnt

    def update(self, target):
        a, b, c, d = self.updateFunction(target)

        bottomMultiplier = 1500.0
        leftRightMultiplier = 100.0
        self.changeBottomThrusterAngle(radians(b))
        self.applyBottomThrust(a * bottomMultiplier)
        self.applyLeftThrust(c * leftRightMultiplier)
        self.applyRightThrust(d * leftRightMultiplier)

        d_target = self.frame.position - target.position
        distanceFit = (1 / (1 + exp(-3.5 + 0.6 * d_target.length)))
        rotationFit = -abs(self.frame.angularVelocity * 0.1)
        absoluteFit = -abs(self.frame.angle * 0.5)
        fitnessAddition = max(0, distanceFit + rotationFit + absoluteFit)
        self.chromosome.fitness += fitnessAddition * 3.0

        self.chromosome.bottomThrustTime.append(a)
        self.chromosome.leftThrustTime.append(c)
        self.chromosome.rightThrustTime.append(d)
        self.chromosome.fitnessTime.append(fitnessAddition)
        self.chromosome.cumulativeFitnessTime.append(self.chromosome.fitness)
        self.chromosome.distanceTime.append(distanceFit)
        self.chromosome.rotationTime.append(rotationFit)
        self.chromosome.absoluteTime.append(absoluteFit)
        self.chromosome.targetX.append(target.position.x)
        self.chromosome.targetY.append(target.position.y)
        self.chromosome.posX.append(self.frame.position.x)
        self.chromosome.posY.append(self.frame.position.y)

    def changeBottomThrusterAngle(self, angle):
        self.bottomThrusterJoint.SetLimits(angle, angle)

    def applyLeftThrust(self, thrust):
        f = self.leftThruster.GetWorldVector(localVector=(0, thrust))
        self.frame.ApplyForce(f, self.leftThruster.worldCenter, True)

    def applyRightThrust(self, thrust):
        f = self.rightThruster.GetWorldVector(localVector=(0, thrust))
        self.frame.ApplyForce(f, self.rightThruster.worldCenter, True)

    def applyBottomThrust(self, thrust):
        f = self.bottomThruster.GetWorldVector(localVector=(0, thrust))
        self.frame.ApplyForce(f, self.bottomThruster.worldCenter, True)

    # Update functions for different algorithms.

    def computeControllerError(self, target):
        dTarget = self.frame.position - target.position
        tAngle = atan2(dTarget.y, dTarget.x) + (pi * 0.5)
        longitudeError = (target.position - self.frame.position).length
        latitudeError = tAngle - self.frame.angle

        if abs(latitudeError) >= (pi / 4):
            latitudeError = -self.frame.angle

        return longitudeError, latitudeError

    def clampOutputs(self, thrust, angle, leftThrust, rightThrust):
        thrust = max(0.0, min(thrust, 1.0))
        leftThrust = max(0.0, min(leftThrust, 1.0))
        rightThrust = max(0.0, min(rightThrust, 1.0))
        angleThrust = max(-30.0, min(angle, 30.0))
        return thrust, angleThrust, leftThrust, rightThrust

    def updatePController(self, target):
        long, lat = self.computeControllerError(target)

        Kp = self.chromosome['Kp']
        C = self.chromosome['C']

        return self.clampOutputs(
            Kp[0] * long + C,
            Kp[1] * -lat,
            Kp[1] * lat,
            Kp[2] * lat)

    def updatePDController(self, target):
        error = target.position.y - self.frame.position.y
        try:
            dError = error - self.lastError
        except AttributeError:
            dError = error

        Kp = self.chromosome['Kp']
        Kd = self.chromosome['Kd']
        C = self.chromosome['C']
        thrust = Kp * error + Kd * dError + C
        thrust = max(0.0, min(thrust, 1.0))

        self.lastError = error

        return thrust, 0.0, 0.0, 0.0

    def updatePIDController(self, target):
        error = target.position.y - self.frame.position.y
        try:
            dError = error - self.lastError
            aError = error + self.lastErrorSum
        except AttributeError:
            self.lastErrorSum = 0.0
            dError = error
            aError = error

        Kp = self.chromosome['Kp']
        Kd = self.chromosome['Kd']
        Ki = self.chromosome['Ki']
        C = self.chromosome['C']
        thrust = Kp * error + Kd * dError + Ki * aError + C
        thrust = max(0.0, min(thrust, 1.0))

        self.lastError = error
        self.lastErrorSum += error

        return thrust, 0.0, 0.0, 0.0

    def updateSHACT(self, target):
        # Angle of the rocket.
        aof = self.frame.angle
        aof = (atan2(sin(aof), cos(aof)) + pi) / (2 * pi)

        # Angle to the target.
        dTarget = self.frame.position - target.position
        att = (atan2(dTarget.y, dTarget.x) + pi) / (2 * pi)

        # Distance to the target.
        dtt = 1 / (1 + exp(4.5 - 0.2 * dTarget.length))

        # Angle of velocity.
        aov = self.frame.linearVelocity
        aov = (atan2(aov.y, aov.x) + pi) / (2 * pi)

        # Magnitude of velocity.
        mov = self.frame.linearVelocity.length
        mov = 1 / (1 + exp(4.4 - 0.4 * mov))

        outputs = self.brain.forward([aof, att, dtt, aov, mov])
        theta = (outputs[1] - 0.5) * 20.0

        return self.clampOutputs(
            outputs[0], theta, outputs[2], outputs[3])

    def updateIslandGA(self, target):
        # Angle of the rocket.
        aof = self.frame.angle
        aof = (atan2(sin(aof), cos(aof)) + pi) / (2 * pi)

        # Angle to the target.
        dTarget = self.frame.position - target.position
        att = (atan2(dTarget.y, dTarget.x) + pi) / (2 * pi)

        # Distance to the target.
        dtt = 1 / (1 + exp(4.5 - 0.2 * dTarget.length))

        # Angle of velocity.
        aov = self.frame.linearVelocity
        aov = (atan2(aov.y, aov.x) + pi) / (2 * pi)

        # Magnitude of velocity.
        mov = self.frame.linearVelocity.length
        mov = 1 / (1 + exp(4.4 - 0.4 * mov))

        outputs = self.brain.forward([aof, att, dtt, aov, mov])
        theta = (outputs[1] - 0.5) * 20.0

        return self.clampOutputs(
            outputs[0], theta, outputs[2], outputs[3])
