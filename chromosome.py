from numpy import (array, cos, exp, random, sin)


class Chromosome():

    def __init__(self, values=None):
        super(Chromosome, self).__init__()

        if values is not None:
            self.values = list(values)
        else:
            self.values = self.randomChromosome()

        self.fitness = 0.0
        self.bottomThrustTime = []
        self.leftThrustTime = []
        self.rightThrustTime = []
        self.fitnessTime = []
        self.cumulativeFitnessTime = []
        self.distanceTime = []
        self.rotationTime = []
        self.absoluteTime = []
        self.targetX = []
        self.targetY = []
        self.posX = []
        self.posY = []

    @classmethod
    def isEvolutionary(cls):
        return True

    @classmethod
    def crossoverRate(cls):
        return 0.8

    @classmethod
    def indexCrossoverRate(cls, index):
        return 0.5

    @classmethod
    def mutationRate(cls, index):
        return 0.01

    @classmethod
    def randomAllele(cls, index):
        return random.rand()

    def randomChromosome(self):
        print('Please override randomChromosome in subclasses.')

    def handleIndex(self, index):
        print('Please override handleIndex in subclasses.')

    def __getitem__(self, index):
        if index == 'left.leg.angle':
            return -1 * ((self.values[0] * 30.0) + 30.0)
        elif index == 'right.leg.angle':
            return 1 * ((self.values[1] * 30.0) + 30.0)
        elif index == 'left.thrust.angle':
            return -1 * ((self.values[2] * 140.0) + 20.0)
        elif index == 'right.thrust.angle':
            return 1 * ((self.values[3] * 140.0) + 20.0)
        elif index == 'height':
            return (self.values[4] * 8) + 3.0
        elif index == 'width':
            return (self.values[5] * 3) + 0.5
        elif index == 'bottom.thrust.angle':
            return (self.values[6] * 80.0) - 40.0
        return self.handleIndex(index)


class SHACTChromosome(Chromosome):

    def __init__(self, values=None):
        self.input, self.hidden, self.output = 5, 10, 4
        super(SHACTChromosome, self).__init__(values)

    def randomChromosome(self):
        values = list(random.rand(1, 8)[0])
        w1 = random.randn(self.hidden, self.input) * 1.0
        b1 = random.randn(self.hidden, 1) * 1.0
        w2 = random.randn(self.output, self.hidden) * 1.0
        b2 = random.randn(self.output, 1) * 1.0
        w1_res = list(w1.reshape([1, -1])[0])
        b1_res = list(b1.reshape([1, -1])[0])
        w2_res = list(w2.reshape([1, -1])[0])
        b2_res = list(b2.reshape([1, -1])[0])
        values = values + w1_res + b1_res + w2_res + b2_res
        return values

    @classmethod
    def randomAllele(cls, index):
        if index >= 8:
            return random.randn()
        else:
            return random.rand()

    def handleIndex(self, index):
        end0 = 8
        end1 = end0 + self.input * self.hidden
        end2 = end1 + self.hidden
        end3 = end2 + self.hidden * self.output
        end4 = end3 + self.output

        if index == 'weights.1':
            return array(self.values[end0:end1]).reshape([self.hidden, -1])
        elif index == 'bias.1':
            return array(self.values[end1:end2]).reshape([self.hidden, -1])
        elif index == 'weights.2':
            return array(self.values[end2:end3]).reshape([self.output, -1])
        elif index == 'bias.2':
            return array(self.values[end3:end4]).reshape([self.output, -1])
        return 0

class IslandChromosome(Chromosome):
    def __init__(self, values=None):
        self.input, self.hidden, self.output = 5, 10, 4
        super(IslandChromosome, self).__init__(values)

    def randomChromosome(self):
        values = list(random.rand(1, 8)[0]) # 8 initial chromosome describing rocket hardware

        # Neural Net weights and biases for a structure of 5 inputs, 10 hidden neurons, 4 output neurons
        w1 = random.randn(self.hidden, self.input) * 1.0
        b1 = random.randn(self.hidden, 1) * 1.0
        w2 = random.randn(self.output, self.hidden) * 1.0
        b2 = random.randn(self.output, 1) * 1.0
        w1_res = list(w1.reshape([1, -1])[0])
        b1_res = list(b1.reshape([1, -1])[0])
        w2_res = list(w2.reshape([1, -1])[0])
        b2_res = list(b2.reshape([1, -1])[0])
        values = values + w1_res + b1_res + w2_res + b2_res
        return values

    @classmethod
    def randomAllele(cls, index):
        if index >= 8:
            return random.randn()
        else:
            return random.rand()

    def handleIndex(self, index):
        end0 = 8
        end1 = end0 + self.input * self.hidden
        end2 = end1 + self.hidden
        end3 = end2 + self.hidden * self.output
        end4 = end3 + self.output

        if index == 'weights.1':
            return array(self.values[end0:end1]).reshape([self.hidden, -1])
        elif index == 'bias.1':
            return array(self.values[end1:end2]).reshape([self.hidden, -1])
        elif index == 'weights.2':
            return array(self.values[end2:end3]).reshape([self.output, -1])
        elif index == 'bias.2':
            return array(self.values[end3:end4]).reshape([self.output, -1])
        return 0

class ControlChromosome(Chromosome):

    def __init__(self, values=None):
        super(ControlChromosome, self).__init__(values)

    @classmethod
    def indexCrossoverRate(cls, index):
        return 0.5 if index >= 8 else 0.0

    @classmethod
    def mutationRate(cls, index):
        return 0.5 if index >= 8 else 0.0

    def randomChromosome(self):
        return [0.5, 0.5, 0.5, 0.5,
            0.8337253258819997, 0.427734004321763,
            0.5, 0.5] + list(random.rand(1, 5)[0])


class PControlChromosome(ControlChromosome):

    def __init__(self, values=None):
        super(PControlChromosome, self).__init__(values)

    def handleIndex(self, index):
        if index == 'Kp':
            return (
                (self.values[8] * 2.0 - 1.0) * 1.0,
                (self.values[9] * 2.0 - 1.0) * 1.0,
                (self.values[10] * 2.0 - 1.0) * 30.0)
        elif index == 'C':
            return self.values[11] * 0.03
        return 0


class PDControlChromosome(ControlChromosome):

    def __init__(self, values=None):
        super(PDControlChromosome, self).__init__(values)

    def handleIndex(self, index):
        if index == 'Kp':
            return self.values[8] * 0.1
        elif index == 'Kd':
            return self.values[9] * 0.1
        elif index == 'C':
            return self.values[10] * 0.1
        return 0


class PIDControlChromosome(ControlChromosome):

    def __init__(self, values=None):
        super(PIDControlChromosome, self).__init__(values)

    def handleIndex(self, index):
        if index == 'Kp':
            return self.values[8] * 0.1
        elif index == 'Kd':
            return self.values[9] * 0.1
        elif index == 'Ki':
            return self.values[10] * 0.1
        elif index == 'C':
            return self.values[11] * 0.1
        return 0
