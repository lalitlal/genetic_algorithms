from pybox2d.examples.framework import (Framework, Keys, main)
from Box2D import (b2CircleShape, b2EdgeShape, b2Filter, b2FixtureDef, b2Vec2)

from chromosome import (PControlChromosome, PDControlChromosome, PIDControlChromosome, SHACTChromosome, IslandChromosome)
from genetic import Genetic
from island_genetic import IslandGenetic
from rocket import Rocket

from numpy import sin


class RocketDemo(Framework):
    name = "Rocket Demo"
    description = "Rocket Ship GA"

    def __init__(self):
        super(RocketDemo, self).__init__()

        self.settings.drawJoints = False
        self.settings.drawMenu = False
        self.settings.drawStats = False
        self.settings.velocityIterations = 3
        self.settings.positionIterations = 1
        self.settings.hz = 20.0
        self.redrawInterval = 0

        # self.world.gravity = b2Vec2(0.0, 0.0)

        self.rockets = []
        # self.genetic = Genetic(
        #     lambda chromosomes: self.startEpoch(chromosomes),
        #     PControlChromosome)

        self.genetic = IslandGenetic(
            lambda chromosomes: self.startEpoch(chromosomes),
            IslandChromosome) # THIS IS FOR ISLAND GENETIC ALGORITHM

        self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-20, -2), (20, -2)]))
        self.target = self.createTarget((0, 60))

    def createTarget(self, position):
        filt = b2Filter(
            categoryBits=Rocket.ROCKET_FILTER,
            maskBits=Rocket.NO_MASK_FILTER)
        frame_fixture = b2FixtureDef(
            shape=b2CircleShape(pos=(0, 0), radius=0.5),
            density=1,
            filter=filt)
        return self.world.CreateStaticBody(
            position=position,
            fixtures=frame_fixture)

    def worldToScreen(self, offset, slope, vector):
        return slope * (vector.x - offset.x), -slope * (vector.y - offset.y)

    # Overridden from the Framework class.
    def Keyboard(self, key):
        if key == Keys.K_a:
            self.genetic.savePlotAndData()
        elif key == Keys.K_q:
            self.redrawInterval += 60
        elif key == Keys.K_w:
            self.redrawInterval -= 60
            self.redrawInterval = max(0, self.redrawInterval)

    def contentStep(self):
        y = 40.0 # + 8.0 * self.genetic.intraEpisodeTime
        x = 0.0 # + 1.0 * self.genetic.intraEpisodeTime * \
            #sin(0.9 * self.genetic.intraEpisodeTime)
        self.target.position = b2Vec2(x, y)
        self.genetic.updateGenetic()
        for rocket in self.rockets:
            rocket.update(self.target)

    # Overridden from the Framework class.
    def Step(self, settings):
        super(RocketDemo, self).Step(settings)
        self.contentStep()

        self.Print('Epoch: {:d}'.format(self.genetic.epoch))
        self.Print('Redraw: {:d}'.format(self.redrawInterval))
        if len(self.genetic.oldChromosomes):
            self.Print('Max Fitness: {:.2f}'.format(
                self.genetic.oldChromosomes[0].fitness))
        offset = self.ConvertScreenToWorld(0.0, 0.0)
        slope = self.ConvertScreenToWorld(1.0, 1.0)
        slope = 1 / (slope.x - offset.x)
        for rocket in self.rockets:
            x, y = self.worldToScreen(offset, slope, rocket.frame.position)
            self.DrawStringAt(x + 20, y,
                'F: {:.1f}'.format(rocket.chromosome.fitness))

        for i in range(self.redrawInterval):
            timeStep = 1.0 / settings.hz
            self.world.Step(timeStep, settings.velocityIterations,
                settings.positionIterations)
            self.world.ClearForces()
            self.contentStep()

    # Callback from Genetic.
    def startEpoch(self, chromosomes):
        for rocket in self.rockets:
            rocket.teardown()
        self.rockets = []
        # IF DEFINE FOR ISLAND MODEL, so we can do it for all chromosomes
        best_rockets = []
        for island in range(len(chromosomes)):
            sorted_island = sorted(chromosomes[island], key=lambda x: x.fitness,
                reverse=True)
            for i in range(10):
                best_rockets.append(sorted_island[i])

        for chromosome in best_rockets:
            self.rockets.append(Rocket(self.world, chromosome))


if __name__ == "__main__":
    main(RocketDemo)

# Appendix at the end of the paper is less than 3 pages of source code
# Most people get the topic difficulty marks
# LaTeX template for formatting
# Don't use long sentences and make it clear and concise
# Know the literature in the field
# Can't just download stuff from Kaggle
# Noveltry is tricky to mark ("the novel idea about this paper is this")
# Results (typically 32% to 36%):
#   How many comparisons, how many techniques, compare against established numbers and expectations
# References (include a lot and don't refer to webpages, REFER TO PAPERS)
# On a team, everyone uploads the exact same paper (only submit 1 PDF file)

# Projectron: A shallow and interpretable network for classifying medical images
# More than 20 references!
