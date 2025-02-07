'''
Running parameter estimation on GW170817 data
'''

import bilby
from gwpy.timeseries import TimeSeries,TimeSeriesDict
from gwpy.plot import Plot
from gwosc import datasets
import matplotlib.pyplot as plt
import pypolychord
from datetime import datetime

# Initialising logger + output directories
logger = bilby.core.utils.logger
outdir0 = 'GW170817Data/'
outdir = outdir0+'gw170817'
label = 'GW170817'
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)

# Used to obtain trigger times from open data (not used)
'''
trigger_time = datasets.event_gps(label)
'''

# Data is from a noise-reduced data set, used in paper https://arxiv.org/pdf/1710.05835
# Data is 2048s long
detectors = ['H1', 'L1', 'V1']

H1data = TimeSeries.read(outdir0+"H1.hdf5",
                       format='hdf5.gwosc')
L1data = TimeSeries.read(outdir0+"L1.hdf5",
                       format='hdf5.gwosc')
V1data = TimeSeries.read(outdir0+"V1.hdf5",
                       format='hdf5.gwosc')

combinedData = TimeSeriesDict()
combinedData['H1'] = H1data
combinedData['L1'] = L1data
combinedData['V1'] = V1data

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

# Strain data for all detectors on one plot
'''
plot = combinedData.plot()
ax = plot.gca()
ax.legend()
ax.set_title("LIGO-Virgo strain data around GW170817")
ax.set_ylabel("Gravitational Wave amplitude [strain]")
ax.axvline(1187008064, color='red', linestyle='--')
ax.set_epoch(1187008064)
plot.show()
'''

# Strain data for all detectors on one figure but 3 subplots, centred on centre time
'''
colour = ['#ee0000','#4ba6ff','#9b59b6']
plot = Plot(combinedDataIfo['H1'],combinedDataIfo['L1'],combinedDataIfo['V1'],separate=True,sharex=True,figsize=(15,8))
plot.text(0.5,0.95,"LIGO-Virgo strain data around GW170817 trigger time",ha='center',fontsize=14)
plot.text(0.04,0.5,"Gravitational Wave amplitude [strain]",va='center',rotation='vertical',fontsize=14)
axes=plot.get_axes()
for i,ax in enumerate(axes):
    ax.axvline(trigger_time, color='black', linestyle='--')
    ax.set_epoch(trigger_time)
    ax.set_ylabel(detectors[i])
    ax.yaxis.set_label_position('right')
    ax.lines[0].set_color(colour[i])
plot.show()
plot.savefig(fname='GW170817/StrainDataTrigger.png')
'''

# Strain data for all detectors on one figure but 3 subplots, around trigger time
'''
colour = ['#ee0000','#4ba6ff','#9b59b6']
plot = Plot(combinedData['H1'],combinedData['L1'],combinedData['V1'],separate=True,sharex=True,figsize=(15,8))
plot.text(0.5,0.95,"LIGO-Virgo strain data around GW170817",ha='center',fontsize=14)
plot.text(0.04,0.5,"Gravitational Wave amplitude [strain]",va='center',rotation='vertical',fontsize=14)
axes=plot.get_axes()
for i,ax in enumerate(axes):
    ax.axvline(1187008064, color='black', linestyle='--')
    ax.set_epoch(1187008064)
    ax.set_ylabel(detectors[i])
    ax.yaxis.set_label_position('right')
    ax.lines[0].set_color(colour[i])
plot.show()
plot.savefig(fname='GW170817/StrainData.png')
'''

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

# Loading data from pickle

ifo_list = bilby.gw.detector.InterferometerList([])
for det in detectors:
   logger.info(f"Loading data for {det}")
   ifo = bilby.gw.detector.Interferometer.from_pickle(filename=f"{outdir0}{det}_data.pkl")
   ifo_list.append(ifo)


# Prior is defined in a local file, GW170817.prior
prior = bilby.gw.prior.BNSPriorDict(filename=outdir0+"GW170817.prior")
deltaT = 0.1
prior['geocent_time'] = bilby.core.prior.Uniform(
    minimum=trigger_time - deltaT / 2,
    maximum=trigger_time + deltaT / 2,
    name="geocent_time",
    latex_label='$t_c$',
    unit='$s$')

# Waveform generator designated. 
# Have not used relative binning in this case
waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments = {
    'waveform_approximant': 'IMRPhenomPv2_NRTidalv2',
    'reference_frequency': 20
    }
)

# Initialising likelihood function
# Distance marginalisation uses a look up table calculated at run time.
# To obtain H0, need P(X| d, cos iota)
# Thus shouldn't marginalise over distance and cos iota.
# iota is theta_jn in bilby (orbital inclination)
# Phase marginalisation is done analytically using a Bessel function. 
# But is formally invalid with precessing waveform such as IMRPhenomPv2
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifo_list,
    waveform_generator,
    priors=prior,
    time_marginalization=False,
    distance_marginalization=False,
    phase_marginalization=True,)

# Sampling
runstarttime = datetime.now()
logger.info(f"Sampling run started at: {runstarttime}")
nlive = 5 # Set to large when actually running on HPC
sampler = 'pypolychord'
result = bilby.run_sampler(
    likelihood,
    prior,
    outdir=outdir,
    label=label,
    sampler=sampler,
    nlive=nlive,
    do_clustering=True,
    save=True,
    use_ratio=True,
    conversion_function=bilby.gw.conversion.generate_all_bns_parameters
    )

logger.info(f"Run completed in: {datetime.now() - runstarttime}")