from math import ceil

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import csv
import os


class DataPersist():

    def __init__(self):
        super(DataPersist, self).__init__()

        self.simStats = {
            'fitness': {
                'best': [],
                'worst': [],
                'total': [],
                'avg': [],
                '25th': [],
                '50th': [],
                '75th': [],
                '90th': []
            },
            'chromosome': {
                'best': [],
                'image': []
            }
        }

    def append(self, chromosomes, totalFitness):
        topVals = chromosomes[0].values
        strTopVals = [str(x) for x in topVals]
        self.simStats['chromosome']['image'].append(topVals)
        self.simStats['chromosome']['best'].append(' '.join(strTopVals))

        self.simStats['fitness']['best'].append(chromosomes[0].fitness)
        self.simStats['fitness']['worst'].append(chromosomes[-1].fitness)
        self.simStats['fitness']['total'].append(totalFitness)
        self.simStats['fitness']['avg'].append(totalFitness / len(chromosomes))

        multiplier = len(chromosomes) if len(chromosomes) > 1 else 0
        per25 = ceil((1 - 0.25) * multiplier)
        per50 = ceil((1 - 0.50) * multiplier)
        per75 = ceil((1 - 0.75) * multiplier)
        per90 = ceil((1 - 0.90) * multiplier)

        self.simStats['fitness']['25th'].append(chromosomes[per25].fitness)
        self.simStats['fitness']['50th'].append(chromosomes[per50].fitness)
        self.simStats['fitness']['75th'].append(chromosomes[per75].fitness)
        self.simStats['fitness']['90th'].append(chromosomes[per90].fitness)

    def savePlotAndData(self, prefix, oldChromosomes):
        # plt.figure(figsize=[6.4, 4.8])
        fig = plt.figure(figsize=(19.2, 9.6))
        gs = GridSpec(2, 3, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[:, -1])

        ax1.plot(self.simStats['fitness']['best'])
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')

        ax2.plot(self.simStats['fitness']['total'], label='Total')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Total Fitness')

        ax3.plot(self.simStats['fitness']['avg'], label='Average')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Average Fitness')

        ax4.plot(self.simStats['fitness']['worst'], label='Worst')
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Worst Fitness')

        ax5.imshow(self.simStats['chromosome']['image'], aspect='auto')
        ax5.set_xlabel('')
        ax5.set_ylabel('')
        ax5.set_xticklabels([])
        ax5.set_yticklabels([])
        ax5.axis('off')
        plt.savefig('data_export/{}_summary.png'.format(prefix), dpi=500)



        fig = plt.figure(figsize=(19.2, 9.6))
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        ax1.plot(oldChromosomes[0].bottomThrustTime, label='Bottom')
        ax1.plot(oldChromosomes[0].leftThrustTime, label='Left')
        ax1.plot(oldChromosomes[0].rightThrustTime, label='Right')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Magnitude')
        ax1.legend()

        ax2.plot(oldChromosomes[0].cumulativeFitnessTime)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Cumulative Fitness')

        ax3.plot(oldChromosomes[0].distanceTime, label='Distance')
        ax3.plot(oldChromosomes[0].rotationTime, label='Angular Velocity')
        ax3.plot(oldChromosomes[0].absoluteTime, label='Angle')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Average Fitness')
        ax3.legend()

        ax4.scatter(oldChromosomes[0].targetX, oldChromosomes[0].targetY, label='Target')
        ax4.scatter(oldChromosomes[0].posX, oldChromosomes[0].posY, label='Rocket')
        ax4.set_xlabel('X (meters)')
        ax4.set_ylabel('Y (meters)')
        ax4.legend()

        plt.savefig('data_export/{}_chromosome.png'.format(prefix), dpi=500)




        if os.path.isfile('data_export/{}_chromosome.csv'.format(prefix)):
            os.remove('data_export/{}_chromosome.csv'.format(prefix))
        with open('data_export/{}_chromosome.csv'.format(prefix), 'w') as f:
            f.write('timeStep,thrustBottom,thrustLeft,thrustRight,fitness,'
                'fitnessCumulative,fitnessDistance,fitnessRotation,'
                'fitnessAbsolute,targetX,targetY,rocketX,rocketY\n')
            for i in range(len(oldChromosomes[0].distanceTime)):
                f.write('{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i,
                    oldChromosomes[0].bottomThrustTime[i],
                    oldChromosomes[0].leftThrustTime[i],
                    oldChromosomes[0].rightThrustTime[i],
                    oldChromosomes[0].fitnessTime[i],
                    oldChromosomes[0].cumulativeFitnessTime[i],
                    oldChromosomes[0].distanceTime[i],
                    oldChromosomes[0].rotationTime[i],
                    oldChromosomes[0].absoluteTime[i],
                    oldChromosomes[0].targetX[i],
                    oldChromosomes[0].targetY[i],
                    oldChromosomes[0].posX[i],
                    oldChromosomes[0].posY[i]))

        if os.path.isfile('data_export/{}_summary.csv'.format(prefix)):
            os.remove('data_export/{}_summary.csv'.format(prefix))
        with open('data_export/{}_summary.csv'.format(prefix), 'w') as f:
            f.write('generation,best,worst,total,avg,25th,50th,75th,90th,'
                'best_chromosome\n')
            for i in range(len(self.simStats['fitness']['best'])):
                f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(i,
                    self.simStats['fitness']['best'][i],
                    self.simStats['fitness']['worst'][i],
                    self.simStats['fitness']['total'][i],
                    self.simStats['fitness']['avg'][i],
                    self.simStats['fitness']['25th'][i],
                    self.simStats['fitness']['50th'][i],
                    self.simStats['fitness']['75th'][i],
                    self.simStats['fitness']['90th'][i],
                    self.simStats['chromosome']['best'][i]))
