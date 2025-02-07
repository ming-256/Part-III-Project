import bilby
import numpy as np
from datetime import datetime
import sys
import scipy
import os
os.environ['TF_USE_LEGACY_KERAS'] = 'True'
import anesthetic
import matplotlib.pyplot as plt
import pandas as pd  
import pypolychord
from pypolychord import PolyChordSettings
import matplotlib as mpl
from matplotlib import rc
from margarine.maf import MAF

# Initialising logger + output directories
logger = bilby.core.utils.logger
outdir0 = 'rds/rds-dirac-dp264-91Bk8EPAcwE/MingYang/GW170817/'
outdir = outdir0+'gw170817'
label = 'GW170817Points2000'

# nDims = Parameters in the posterior i.e. BNS merger model
nDims = 16
samples = anesthetic.read_chains(outdir+'/chains/'+label)
samples = pd.concat([samples.iloc[:, :nDims], samples.iloc[:, -3:]], axis=1)

# Plot posteriors
fs = 13
column_names = ['mass_ratio', 'chirp_mass', 'luminosity_distance', 'cos_theta_jn']
fig, axes = anesthetic.make_2d_axes(column_names, upper=False)
samples.plot_2d(axes, alpha=0.8, label='Posterior samples', levels=[0.99994, 0.99730, 0.95450, 0.68269])
axes.iloc[0,0].legend(fontsize=fs)

column_names = ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl']
fig, axes = anesthetic.make_2d_axes(column_names, upper=False)
samples.plot_2d(axes, alpha=0.8, label='Posterior samples', levels=[0.99994, 0.99730, 0.95450, 0.68269])
axes.iloc[0,0].legend(fontsize=fs)

# Plotting posterior and joint distribution for D_L and cos_theta_jn
column_names = ['luminosity_distance', 'cos_theta_jn']
fig, axes = anesthetic.make_2d_axes(column_names, upper=False)
samples.plot_2d(axes, alpha=0.8, label='Posterior samples', levels=[0.99994, 0.99730, 0.95450, 0.68269])
axes.iloc[0,0].legend(fontsize=fs)

# Train a normalizing flow (margarine) on these samples
# using the sample weights from polychord:

trigger_time = 1187008882.43
prior = bilby.gw.prior.BNSPriorDict(filename=outdir0+"GW170817.prior")
deltaT = 0.1
prior['geocent_time'] = bilby.core.prior.Uniform(
    minimum=trigger_time - deltaT / 2,
    maximum=trigger_time + deltaT / 2,
    name="geocent_time",
    latex_label='$t_c$',
    unit='$s$')

data = samples.iloc[:, :nDims]
data['geocent_time'] = data['geocent_time'] - trigger_time #need this step as geocent time is GPS time, which to float32 precision is same for all samples
data = data.values
weights = samples.get_weights()
#theta_max and theta_min are set to the prior values - where these are 0 or 1, we need to set to slightly different values, otherwise will give nans when we evaluate
bij = MAF(data, weights=weights)
bij.train(20000, early_stop=True)
bij.save(f'{outdir}/MAF_{label}.pkl')

##if want to train on equal weight samples instead (sometimes margarine does better at this):
#data = samples.compress().iloc[:, :nDims]
#data['geocent_time'] = data['geocent_time'] - injection_parameters['geocent_time']  #need this step as geocent time is GPS time, which to float32 precision is same for all samples
#data = data.values
#weights = np.ones(len(data))
##rest is same as above

#now let's plot the flow and see how well it's learned the posteriors

from anesthetic import MCMCSamples
columns = list(samples.columns.get_level_values(0))
x = bij(np.random.uniform(size=(10000, nDims)))
maf_samples = MCMCSamples(data=x, weights=np.ones(len(x)))
maf_samples.rename(columns={i: columns[i] for i in range(nDims)}, inplace=True)
maf_samples = maf_samples.astype({'geocent_time': 'float64'})
maf_samples['geocent_time'] = maf_samples['geocent_time'] + trigger_time#get actual merger time again

column_names =['mass_ratio', 'chirp_mass', 'luminosity_distance', 'theta_jn']
fig, axes = anesthetic.make_2d_axes(column_names, upper=False)
samples.plot_2d(axes, alpha=0.8, label='Original', levels=[0.99994, 0.99730, 0.95450, 0.68269])
maf_samples.plot_2d(axes, alpha=0.5, label='MAF', levels=[0.99994, 0.99730, 0.95450, 0.68269])
axes.iloc[0, 0].legend()

column_names = ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl']
fig, axes = anesthetic.make_2d_axes(column_names, upper=False)
samples.plot_2d(axes, alpha=0.8, label='Original', levels=[0.99994, 0.99730, 0.95450, 0.68269])
maf_samples.plot_2d(axes, alpha=0.5, label='MAF', levels=[0.99994, 0.99730, 0.95450, 0.68269])
axes.iloc[0, 0].legend()

#might also want to check how well maf has learned the NS probabilities
data = samples.iloc[:, :nDims]#.values
data['geocent_time'] = data['geocent_time'] - trigger_time
data = data.values
weights = samples.get_weights()
log_pitilde_maf = MAF.log_prob(bij, data).numpy()

params = {"mass_ratio": 0, "chirp_mass": 0, "luminosity_distance": 0, "dec": 0, "ra": 0, "theta_jn": 0, "psi": 0, "phase": 0, "a_1": 0, "a_2": 0, "tilt_1": 0, "tilt_2": 0, "phi_12": 0, "phi_jl": 0, "geocent_time": 0}
log_prior = []
for i in range(len(samples)):
   params.update(zip(params, samples.iloc[i, :nDims]))
   log_prior.append(prior.ln_prob(params))
log_posterior = samples.logL.values + log_prior - samples.logZ()


iter_start = 5000#play around with this to see how flow deteriorates in quality in the tails of the distribution
iter_end = len(log_posterior)
NS_posterior = log_posterior[iter_start:iter_end]
plt.plot(NS_posterior, log_pitilde_maf[iter_start:iter_end], 'o', alpha=0.5, label='MAF')
#if flow is perfect, all the points should lie on the y=x line
plt.plot(np.linspace(NS_posterior.min(), NS_posterior.max(), 100), np.linspace(NS_posterior.min(), NS_posterior.max(), 100), 'k--', alpha=0.8)
plt.xlabel('log_posterior from NS')
plt.ylabel('log_posterior from flow')
plt.legend()