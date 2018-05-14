import numpy as np
import operator
from functools import partial
from matplotlib import pyplot as plt
import csv
from itertools import izip
import sys

# TODO: generate realistic examples

# TODO: RandomPulse below is the only MetricGenerator that accepts a callable
# where a random parameter generator can be supplied. This behavior may be
# desirable for other metric generators too.


# globals
T = None    # time stamps
lT = 0      # length of period

# utils
valof = lambda arg: arg() if callable(arg) else arg
def filepathnametitle(fpath, fname): 
    return ("%s/%s" % (fpath, fname), fname) if fpath else (None, fname)

# generic metric generator

class MetricGenerator(object):
    # super class to share gen()
    def __init__(self):
        pass
    # subclasses should populate this method 
    # to provide a way to generate a single
    # data point from the model
    def next():
        pass
    # use this function to generate historical data
    # Note: needs to be vectorized for efficiency
    def gen(self, N=1):
        s = []
        for n in range(N):
            s.append(self.next())
        return s

# model

class Model(MetricGenerator):
    # Additive and multiplicative models are the most common ways 
    # to generate complex metrics out of simple waveforms
    def __init__(self, op=operator.add, bias=0):
        # TODO: check if op is either ADD or MULT
        self.op = op
        self.bias = bias
        self.models = []
    # Add a model to the ensemble with its weight
    def add(self, model, weight):
        self.models.append([model, weight])
    # reset bias
    def resetBias(self, bias):
        self.bias = bias
    # Query the next data point
    def next(self):
        val = self.bias
        for model, weight in self.models:
            val = self.op(val, valof(weight) * model.next())
        return val

# Markov Chain for generating anomalous data

class MarkovChain(MetricGenerator):
    # STM: State Transition Matrix, list of lists
    def __init__(self, STM, initState=0, resetBias=False):
        self.STM = STM
        self.state = initState
        self.resetBias = resetBias
        self.models = []
        self.val = None  # last emitted value
        self.labels = []
    # Add an emission model for a state
    def add(self, model):
        self.models.append(model)
    # Query the next data point
    def next(self):
        # decide state transition
        newState = np.argmax(np.random.multinomial(1, self.STM[self.state], 1))
        self.labels.append(newState)
        if self.state != newState:
            self.state = newState
            if self.resetBias:
                self.models[self.state].resetBias(self.val)
        self.val = self.models[self.state].next()
        return self.val
    # Query the labels so far
    def getLabels(self):
        return self.labels
    
# 1D noise

class Noise(MetricGenerator):
    # Define a noise model out of a random generator.
    # If the model type is "normal", expected parameters are mean and stddev.
    # Default is uniform with no parameters.
    def __init__(self, modelType = "uniform", modelParams = None):
        def model(n):
            if modelType == "normal":
                return np.random.normal(modelParams[0], modelParams[1], n)
            else:
                return np.random.rand(n, 1)
        self.model = model
    def next(self):
        return self.model(1)[0]    # lift the value out of the returned array

# Step

class Step(MetricGenerator):
    # Simple deterministic step function
    def __init__(self, onset, initT=0, dT=1):
        self.onset = onset
        self.dT = dT
        self.t = initT
    def next(self):
        val = 0 if self.t < self.onset else 1
        self.t = self.t + self.dT
        return val

class RandomStep(Step):
    # Simple step with random onset. 
    # Assumes length of the time series is known 
    # - i.e. for generating historical data.
    # Set onset time to random fraction of 
    # the total time series length lT.
    def __init__(self, lT):
        self.onset = np.random.rand() * lT
        super(RandomStep, self).__init__(self.onset)
        
class RandomStatefulStep(MetricGenerator):
    # Simple step with random onset.
    # This version uses a random state transition,
    # hence does not require the length of the data.
    # STP (state transition probability) of 0.999 should 
    # yield a random step around 1000th data point.
    def __init__(self, STP=0.999):
        self.STP = STP
        self.state = 0
    def next(self):
        if self.state == 0:
            if np.random.rand() > self.STP:
                self.state = 1
        return self.state

# Ramp

class Ramp(MetricGenerator):
    # Simple deterministic ramp function
    def __init__(self, onset, duty, slope=1, initT=0, dT=1):
        self.onset = onset
        self.reset = onset + duty
        self.slope = slope
        self.dT = dT
        self.t = initT
        self.val = 0
    def next(self):
        if self.t < self.onset:
            self.val = 0
        elif self.t < self.reset:
            self.val = self.val + self.slope
        else:
            self.val = self.val
        self.t = self.t + self.dT
        return self.val

class RandomRamp(Ramp):
    # Simple ramp with random onset and duty
    # This version requires knowing the time series length
    # i.e. use for historical data
    def __init__(self, lT):
        self.onset = np.random.rand() * lT
        self.duty = np.random.rand() * lT
        self.slope = 2*np.random.rand() - 1
        super(RandomRamp, self).__init__(self.onset, self.duty, self.slope)

class RandomStatefulRamp(MetricGenerator):
    # Simple random ramp using states to 
    # represent onset time and duty.
    # STP (state transition probability)
    # TODO: Make this a function of t.
    def __init__(self, onsetSTP=0.5, dutySTP=0.5, slope=1):
        self.onsetSTP = onsetSTP
        self.dutySTP = dutySTP
        self.slope = slope
        self.state = 0
        self.val = 0
    def next(self):
        if self.state == 0:
            if np.random.rand() > self.onsetSTP:
                self.state = 1
        elif self.state == 1:
            if np.random.rand() > self.dutySTP:
                self.state = 2
        if self.state == 0:
            self.val = 0
        elif self.state == 1:
            self.val = self.val + self.slope
        else:
            self.val = self.val
        return self.val

class Trend(Ramp):
    # Trend is ramp that starts immediately and lasts forever
    def __init__(self, slope=1):
        super(Trend, self).__init__(0, np.inf)

# Pulse train

class Pulse(MetricGenerator):
    # Single pulse to be chained into a train - i.e. a helper
    def __init__(self, onset, duty, initT=0, dT=1):
        self.onset = onset
        self.offset = self.onset + duty
        self.dT = dT
        self.t_ = initT
        self.state = 0
    def next(self):
        if self.t_ < self.onset:
            self.state = 0
        elif self.t_ < self.offset:
            self.state = 1
        else:
            self.state = -1
        self.t_ = self.t_ + self.dT
        return self.state
    def getT(self):
        return self.t_
    
class PulseTrain(MetricGenerator):
    # (Periodic) Pulse Train
    def __init__(self, onset, duty, initT=0, dT=1):
        self.onset = onset
        self.duty = duty
        self.dT = dT
        self.t = initT
        self.pulse = Pulse(self.onset, self.duty, initT=self.t)
    def _newPulse(self):
        return Pulse(self.t + self.onset, self.duty, initT=self.t)
    def next(self):
        val = self.pulse.next()
        if val == -1:
            self.pulse = self._newPulse()
            val = self.pulse.next()
        self.t = self.t + self.dT
        return val

class RandomPulses(MetricGenerator):
    # Random Pulses
    def __init__(self, meanOnset, meanDuty, initT=0, dT=1):
        self.meanOnset = meanOnset
        self.meanDuty = meanDuty
        self.dT = dT
        self.t = initT
        self.pulse = self._newPulse()
    def _newPulse(self):
        return Pulse(self.t + np.random.poisson(int(valof(self.meanOnset))) + self.dT, 
                    np.random.poisson(int(valof(self.meanDuty))) + self.dT, 
                    initT=self.t)
    def next(self):
        val = self.pulse.next()
        if val == -1:
            self.pulse = self._newPulse()
            val = self.pulse.next()
        self.t = self.t + self.dT
        return val

# helpers

def set_time_axis(t_start=0, t_end=3600, dt=1):
    global T, lT
    T = np.arange(t_start, t_end, dt)   # time axis
    lT = len(T)                         # number of data points in time series

def plot(T, V, L=None, saveTo=None, title=""):
    f = plt.figure()
    plt.plot(T, V, color="blue")
    if L:
        plt.hold(True)
        plt.plot(T, L, color="red")
    if saveTo:
        plt.savefig(saveTo)
    else:
        plt.title(title)
        plt.show()
    plt.close()
    
def save(T, V, L=None, saveTo=None):
    if saveTo:
        with open("%s.csv" % saveTo, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(izip(T, V, L) if L else izip(T, V))
    
def plot_save(T, V, L=None, target=(None, "")):
    plot(T, V, L, target[0], target[1])
    save(T, V, L, target[0])

def AD_examples(filenametitle=lambda x:(None, x)):

    # np.random.seed(1)
    set_time_axis()

    # a CPU example

    STM = [[.001, .999], [.999, .001]] # set-1 : 349 anomalies out of 3600
    mc = MarkovChain(STM)
    # Normal values
    model0 = Model()
    # Base
    model0.add(Noise("normal", [0.4,.2]), 1.)
    # noise
    model0.add(RandomPulses(np.random.rand()*lT/15., 0), lambda: np.random.rand() * .1)
    mc.add(model0)
    # Anomaly
    model1 = Model()
    # Base
    model1.add(Noise("normal", [0.6,.1]), 1.)
    # Noise
    model1.add(RandomPulses(np.random.rand()*lT/15., 0), lambda: np.random.rand() * .1 + 2.8 )
    mc.add(model1)
    #
    V = mc.gen(lT)
    L = mc.getLabels()
    plot_save(T, V, L, target=filenametitle("CPU_example"))

def run_all(filepath=None):
    filenametitle = partial(filepathnametitle, filepath)
    AD_examples(filenametitle)
    
if __name__ == "__main__":
    filepath = None
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    run_all(filepath)

