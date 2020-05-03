from data_persist import DataPersist

from numpy import random
from random import uniform
import numpy


class IslandGenetic():

    def __init__(self, newEpochCallback, chromosomeClass):
        super(IslandGenetic, self).__init__()

        self.epoch = 0
        self.chromosomes = []
        self.islands = []
        self.oldChromosomes = []
        self.chromosomeClass = chromosomeClass
        self.intraEpisodeTime = 0.0
        self.newEpochCallback = newEpochCallback
        self.dataPersist = DataPersist()

        if chromosomeClass.isEvolutionary():
            self.numChromosomes = 100
            self.numKeepChromosomes = 6
        else:
            self.numChromosomes = 1
            self.numKeepChromosomes = 1

        self.firstEpoch()

    @classmethod
    def migrationFrequency(cls):
        return 50 #how many generations between migration

    @classmethod
    def migrationRate(cls):
        return 2 #numer of chromosomes migrating to each population

    @classmethod
    def islandPopulation(cls):
        return 100

    @classmethod
    def numIslands(cls):
        return 10

    def updateGenetic(self):
        self.intraEpisodeTime += 1 / 20.0
        if self.intraEpisodeTime > 15.0:
            self.intraEpisodeTime = 0.0
            self.newEpoch()

    def crossover(self, parent1, parent2):
        child1 = self.chromosomeClass(parent1.values)
        child2 = self.chromosomeClass(parent2.values)
        if uniform(0, 1) < self.chromosomeClass.crossoverRate():
            values1 = parent1.values
            values2 = parent2.values
            for i in range(len(values1)):
                if uniform(0, 1) < self.chromosomeClass.indexCrossoverRate(i):
                    child1.values[i] = values2[i]
                    child2.values[i] = values1[i]
        return child1, child2

    def mutate(self, chromosome):
        mutated = self.chromosomeClass(chromosome.values)
        for i in range(len(chromosome.values)):
            if uniform(0, 1) < self.chromosomeClass.mutationRate(i):
                mutated.values[i] = self.chromosomeClass.randomAllele(i)
        return mutated

    def migrate(self, index):
        #sort migration population by maximum fitness
        migratorPop = sorted(self.chromosomes[index], key=lambda x: x.fitness,
            reverse=True)

        for k in range(self.numIslands()):
            if k != index:
                #receiving population sorted by least fit
                receivePop = sorted(self.chromosomes[k], key=lambda x: x.fitness, reverse=False)
                #fitness of migration population is less than least fit of receiving
                if migratorPop[0].fitness <= receivePop[0].fitness:
                    # print("sendpop", migratorPop[0].fitness, "recievepop", receivePop[0].fitness)
                    continue
                else:
                    for i in range(self.migrationRate()):
                        #print("receivepop: ", receivePop[i].fitness, "sendpop: ", migratorPop[i].fitness)
                        receivePop[i] = migratorPop[i]
                    self.chromosomes[k] = receivePop
        return


    def computeDistribution(self, chromosomes, island_num):
        totalFitness = 0.0
        for chromosome in chromosomes:
            totalFitness += chromosome.fitness
        if totalFitness == 0.0:
            totalFitness = 0.01

        distribution = []
        sumOfProbabilities = 0.0
        for chromosome in chromosomes:
            probability = chromosome.fitness / totalFitness
            distribution.append(sumOfProbabilities + probability)
            sumOfProbabilities += probability

        self.dataPersist.append(chromosomes, totalFitness)

        print('Epoch {:3d} Island{:3d} Max {:.2f} Total {:.2f}'.format(self.epoch,
            island_num, chromosomes[0].fitness, totalFitness))
        return distribution

    def selectParentFromDistribution(self, distribution, chromosomes):
        selection = uniform(0, 1)
        for i, prob in enumerate(chromosomes[1:]):
            if selection < distribution[i]:
                return chromosomes[i - 1]
        return chromosomes[0]

    def firstEpoch(self):
        for k in range(self.numIslands()):
            self.chromosomes.append([])
            for i in range(self.numChromosomes):
                self.chromosomes[k].append(self.chromosomeClass())
        self.newEpochCallback(self.chromosomes)

    def newEpoch(self):
        self.epoch += 1

        for k in range(self.numIslands()):
            if self.epoch % self.migrationFrequency() == 0:
                self.migrate(k) #migrate from this subpopulation to all others

            self.oldChromosomes = sorted(self.chromosomes[k], key=lambda x: x.fitness,
                reverse=True)
            distribution = self.computeDistribution(self.oldChromosomes, k)

            self.chromosomes[k] = []
            for i in range(self.numKeepChromosomes):
                self.chromosomes[k].append(self.chromosomeClass(
                    self.oldChromosomes[i].values))

            while (len(self.chromosomes[k]) < self.numChromosomes):
                parent1 = self.selectParentFromDistribution(distribution,
                    self.oldChromosomes)
                parent2 = self.selectParentFromDistribution(distribution,
                    self.oldChromosomes)
                child1, child2 = self.crossover(parent1, parent2)
                self.chromosomes[k].append(self.mutate(child1))
                self.chromosomes[k].append(self.mutate(child2))

        self.newEpochCallback(self.chromosomes)

    def savePlotAndData(self):
        self.dataPersist.savePlotAndData(str(self.chromosomeClass.__name__),
            self.oldChromosomes)
