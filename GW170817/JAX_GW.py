# Prerequisites: 
# pip install anesthetic
# pip install git+https://git.ligo.org/lscsoft/ligo-segments.git
# pip install git+https://github.com/ming-256/jim-load-file
# pip install git+https://github.com/handley-lab/blackjax@proposal
#

import blackjax
import blackjax.ns.adaptive
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from anesthetic import NestedSamples, read_chains
from jimgw.prior import (
    CombinePrior,
    CosinePrior,
    PowerLawPrior,
    UniformPrior,
    SinePrior
)
from jimgw.single_event.likelihood import TransientLikelihoodFD, HeterodynedTransientLikelihoodFD
from jimgw.single_event.transforms import (
    MassRatioToSymmetricMassRatioTransform,
)
from jimgw.single_event.waveform import RippleIMRPhenomD_NRTidalv2

from jimgw.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
)
from jimgw.transforms import BoundToUnbound

from gwpy.timeseries import TimeSeries,TimeSeriesDict
import bilby
from jimgw.single_event.utils import Mc_q_to_m1_m2

jax.config.update("jax_enable_x64", True)
filename = 'JAXGW170817'

# | Fetch ligo data
detectors = ['H1', 'L1', 'V1']
outdir0 = ''

H1data = TimeSeries.read(outdir0+"H1.hdf5",
                       format='hdf5.gwosc')
L1data = TimeSeries.read(outdir0+"L1.hdf5",
                       format='hdf5.gwosc')
V1data = TimeSeries.read(outdir0+"V1.hdf5",
                       format='hdf5.gwosc')

trigger_time = 1187008882.43
centre_time = 1187008064 # Centre time of the data file

sample_rate = 4096 # Using the 4096Hz data. 16384Hz also exists
duration = 128  # Length of signal
post_trigger_duration = 2  # Time between trigger time and end of signal
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

data_duration = 2048 # Using the entire data set for psd
psd_start_time = centre_time - data_duration / 2
psd_end_time = start_time

maximum_frequency = 2048 # Interferometer frequency
minimum_frequency = 23
roll_off = 0.4 # Roll off duration of tukey window in seconds, default is 0.4s.

# Cropped data around the trigger time for ifo and psd
combinedDataPsd = TimeSeriesDict()
combinedDataPsd['H1'] = H1data.crop(psd_start_time,psd_end_time)
combinedDataPsd['L1'] = L1data.crop(psd_start_time,psd_end_time)
combinedDataPsd['V1'] = V1data.crop(psd_start_time,psd_end_time)

combinedDataIfo = TimeSeriesDict()
combinedDataIfo['H1'] = H1data.crop(start_time,end_time)
combinedDataIfo['L1'] = L1data.crop(start_time,end_time)
combinedDataIfo['V1'] = V1data.crop(start_time,end_time)

'''
# Creating the interferometer list from the analysis and psd data and pickling
ifo_list = bilby.gw.detector.InterferometerList([])
for det in detectors:
    logger.info("Loading analysis data for ifo {}".format(det))
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    ifo.strain_data.set_from_gwpy_timeseries(combinedDataIfo[det])

    logger.info("Loading psd data for ifo {}".format(det))
    psd_alpha = 2 * roll_off / duration
    psd = combinedDataPsd[det].psd(
        fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median"
    )
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value
    )
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency
    ifo_list.append(ifo)
    logger.info(f"Pickling data for {det}")
    ifo.to_pickle(label=f"{det}_data",outdir=outdir0)

logger.info("Saving data plots to {}".format(outdir))
ifo_list.plot_data(outdir=outdir0, label=label)
'''

from jimgw.single_event.detector import H1, L1, V1
H1.load_data_from_file(combinedDataIfo['H1'], combinedDataPsd['H1'],duration, roll_off, minimum_frequency, maximum_frequency)
L1.load_data_from_file(combinedDataIfo['L1'], combinedDataPsd['L1'],duration, roll_off, minimum_frequency, maximum_frequency)   
V1.load_data_from_file(combinedDataIfo['V1'], combinedDataPsd['V1'],duration, roll_off, minimum_frequency, maximum_frequency)  

# | Define the waveform
transforms = [
    MassRatioToSymmetricMassRatioTransform,
]
waveform = RippleIMRPhenomD_NRTidalv2(f_ref=minimum_frequency, use_lambda_tildes=True)

# | Define the prior
prior = CombinePrior(
    [
        UniformPrior(1.18, 1.21, parameter_names=["M_c"]),
        UniformPrior(0.125, 1.0, parameter_names=["q"]),
        UniformPrior(-1.0, 1.0, parameter_names=["s1_z"]),
        UniformPrior(-1.0, 1.0, parameter_names=["s2_z"]),
        SinePrior(parameter_names=["iota"]),
        PowerLawPrior(1.0, 75.0, 2.0, parameter_names=["d_L"]),
        UniformPrior(-0.05, 0.05, parameter_names=["t_c"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"]),
        UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
        UniformPrior(1.0, 100.0, parameter_names=["H_0"]),
        UniformPrior(-1000.0, 1000.0, parameter_names=["v_p"]),
        UniformPrior(0.0, 1000.0, parameter_names=["lambda_tilde"]),
        UniformPrior(-5000.0, 5000.0, parameter_names=["delta_lambda_tilde"]),
    ]
)
# H0 should be a log uniform prior !
# Log uniform is not a function in jimgw, but it is in bilby.
# I can try implement this in jimgw.prior ?

labels = {
    "M_c": r"$M_c$",
    "q": r"$q$",
    "s1_z": r"$s_{1z}$",
    "s2_z": r"$s_{2z}$",
    "iota": r"$\iota$",
    "d_L": r"$d_L$",
    "t_c": r"$t_c$",
    "phase_c": r"$\phi_c$",
    "psi": r"$\psi$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
    "H_0": r"$H_0$",
    "v_p": r"$v_p$",
    "lambda_tilde": r"$\tilde{\Lambda}$",
    "delta_lambda_tilde": r"$\Delta\tilde{\Lambda}$",
}

likelihood = HeterodynedTransientLikelihoodFD(
    [H1, L1, V1],
    waveform=waveform,
    # n_bins=301,
    trigger_time=trigger_time,
    duration=duration,
    post_trigger_duration=post_trigger_duration,
    prior=prior,
    # sample_transforms=sample_transforms,
    likelihood_transforms=transforms
)
exit()

vals, ravel = jax.flatten_util.ravel_pytree(
    prior.sample(jax.random.PRNGKey(0), 1)
)


def logprior_fn(x):
    folded_coords = jax.tree_util.tree_map(lambda y: y.squeeze(), ravel(x))

    logprob = prior.log_prob(folded_coords)
    iota_out_of_bounds = (folded_coords["iota"] < -1.0) | (folded_coords["iota"] > 1.0)
    logprob = jnp.where(iota_out_of_bounds, -jnp.inf, logprob)
    dec_out_of_bounds = (folded_coords["dec"] < -jnp.pi / 2.0) | (
        folded_coords["dec"] > jnp.pi / 2.0
    )
    logprob = jnp.where(dec_out_of_bounds, -jnp.inf, logprob)
    return logprob


@jax.jit
def ll(x):
    ll_vr = jax.scipy.stats.norm.logpdf(3327, x['v_p']+x['H_0']*x['d_L'], 72)

    ll_vp = jax.scipy.stats.norm.logpdf(310, x['v_p'], 150)

    return likelihood.evaluate(
        jax.tree_util.tree_map(
            lambda y: y.squeeze(), transforms[0].forward(ravel(x))
        ),
        None,
    ) + ll_vr + ll_vp



# | Define the Nested Sampling algorithm
n_dims = len(prior.parameter_names)
n_live = 1000
n_delete = 200
num_mcmc_steps = n_dims * 5

# | Initialize the Nested Sampling algorithm
nested_sampler = blackjax.ns.adaptive.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=ll,
    n_delete=n_delete,
    num_mcmc_steps=num_mcmc_steps,
)


@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = nested_sampler.step(subk, state)
    return (state, k), dead_point


# | Sample live points from the prior
rng_key = jax.random.PRNGKey(0)
rng_key, init_key = jax.random.split(rng_key, 2)
initial_particles = prior.sample(init_key, n_live)
initial_particles = jnp.vstack(jax.tree_util.tree_flatten(initial_particles)[0]).T
state = nested_sampler.init(initial_particles, ll)


# | Run Nested Sampling
dead = []
with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
        while not state.sampler_state.logZ_live - state.sampler_state.logZ < -3:
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(n_delete)  # Update progress bar

# | anesthetic post-processing
dead = jax.tree.map(lambda *args: jnp.concatenate(args), *dead)
live = state.sampler_state
logL = np.concatenate((dead.logL, live.logL), dtype=float)
logL_birth = np.concatenate((dead.logL_birth, live.logL_birth), dtype=float)
data = np.concatenate((dead.particles, live.particles), dtype=float)
columns = ravel(vals).keys()
samples = NestedSamples(
    data, logL=logL, logL_birth=logL_birth, columns=columns, labels=labels
)
samples.to_csv(filename)

logzs = samples.logZ(100)
print(f"{logzs.mean()} +- {logzs.std()}")



# | Plot the results
'''
# | For post-processing
# samples = read_chains(filename)
params = [
    "M_c",
    "q",
    # "s1_z",
    # "s2_z",
    # "iota",
    # "d_L",
    # "t_c",
    # "phase_c",
    # "psi",
    "ra",
    "dec",
]
samples.plot_2d(params)

'''