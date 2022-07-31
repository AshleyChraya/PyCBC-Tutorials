import numpy as np
from astropy.utils.data import download_file
from pycbc.catalog import Catalog, Merger
from pycbc.distributions import JointDistribution, SinAngle, Uniform
from pycbc.filter import highpass, resample_to_delta_t
from pycbc.frame import read_frame
from pycbc.inference import models, sampler
from pycbc.psd import interpolate, inverse_spectrum_truncation


# 1D Gaussian
def gauss(x, mean, sigma):
    return np.exp((-((x - mean) ** 2)) / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)


# All the Model names in the model module
def model_names():
    for i in models.models:
        print(i)


# Information about catalogs, events
def info():
    m = Catalog(source="gwtc-3")  # Returns an object
    print("Events are : ")
    for i in m:
        print(i)
    # Returns dictionary with names and info website link about the event
    merger_events = m.mergers
    # Names of all the events in the given catalog
    print("Event names : ", merger_events.keys())
    # Returns just names # (Use above .keys method or this)
    merger_names = m.names
    # Returns information about all events # Dictionary format, where keys are events name
    merger_details = m.data
    print(merger_details)
    # Returns parameter value for all events
    parameters = m.median1d("distance")
    # use any parameter Example mchirp = m.median1d('mchirp')

    d = Merger("GW170817")
    event_name = d.common_name
    # data about the event # Dictionary where keys are parameters that can be used in median1d
    event_detail = d.data
    print("Details about the event : ", event_detail)
    frame = d.frame  # Not sure about this functionality????
    print("Frame of the event is ", frame)
    t = d.time  # time of the event
    print("TIme of the event occuring : ", t)
    par = d.median1d("distance")  # Returns parameter value
    redshift = d.median1d("redshift")
    # By default above returns parameters in the *source* frame.
    par = par * (1 + redshift)
    #  Converting it into detector frame by * (1 + z)

    return event_name, event_detail, frame, t, par


# GW signal processing: 1. Passing through high pass filter 2. Whitening
def gw_data(detector, event_name):
    # Download the gravitational wave data for GW170817
    url = "https://dcc.ligo.org/public/0146/P1700349/001/{}-{}1_LOSC_CLN_4_V1-1187007040-2048.gwf"
    # Downloading "H-H1_LOSC_4_V2-1128678884-32.gwf" file
    fname = download_file(url.format(detector[0], detector[0]), cache=True)

    m = Merger(event_name)
    # read_frame(file_name, channel_name, start, end); channels names is typically IFO:LOSC-STRAIN, where IFO can be H1/L1/V1
    data = read_frame(fname, "{}:LOSC-STRAIN".format(detector), start_time=int(m.time - 260), end_time=int(m.time + 40))
    # Read the data directly from the Gravitational-Wave Frame (GWF) file. or
    # this data can be obtained by m = Merger("GW170817"); data = m.strain("H1")
    print("Sample rate of the downloaded file is ", data.sample_rate)

    # Signal processing; This data is colored gaussian and not whitened gaussian. We need to get flat and whitened gaussian noise.
    # Colored gaussian: different power at different frequencies.("Noise properties" different at different freq)
    # Whitening takes the data and attempts to make the power spectral density flat, so that all frequencies contribute equally.
    # Remove low frequency bec noise of the instrument is at low freq
    data = highpass(data, 15.0)
    # Low pass filter removes high freq
    # However, after removing low freq there is clearly still some dominant frerquencies.
    # To equalize this, we would need to apply a whitening filter.

    # This produces a whitened set.
    # This works by estimating the power spectral density from the
    # data and then flattening the frequency response.
    # (1) The first option sets the duration in seconds of each
    #     sample of the data used as part of the PSD estimate.
    # (2) The second option sets the duration of the filter to apply

    data = data.whiten(4, 4)

    # Resample data to 2048 Hz # Downsample to lower sampling rate # WHY???
    data = resample_to_delta_t(data, 1.0 / 2048)
    print("Downsampled to a lowering rate ", data.sample_rate)

    # Limit to times around the signal
    data = data.time_slice(m.time - 112, m.time + 16)
    # m.time = time of collision; m.start_time = start time of observation of data; m.end_time = end time of observation of data;
    # m.duration = duration of observation; m.sample_times = time array at the time of observation of each data point (used for plotting
    # timeseries strain) ; m.time_slice(start, end)

    # Convert to a frequency series by taking the data's FFT
    data_freq = data.to_frequencyseries()

    return data, data_freq


# Power spectral density of the time series data
def PSD(data):
    # Estimate the power spectral density of the data
    # This estimates the PSD by sub-dividing the data into 4s long segments. (See Welch's method)
    psd = data.psd(4)

    # Now that we have the psd we need to interpolate it to match our data
    # and then limit the filter length of 1 / PSD. After this, we can
    # directly use this PSD to filter the data in a controlled manner
    psd = interpolate(psd, data.delta_f)

    # 1/PSD will now act as a filter with an effective length of 4 seconds
    # Since the data has been highpassed above 15 Hz, and will have low values
    # below this we need to informat the function to not include frequencies
    # below this frequency.
    psd = inverse_spectrum_truncation(psd, int(4 * psd.sample_rate), trunc_method="hann", low_frequency_cutoff=20.0)
    return psd


# Prior; Priors are on parameters which will vary, use joint distribution
def prior_func(parameters):
    # sourcery skip: inline-immediately-returned-variable
    # Priors on the parameters which are varied
    inclination_prior = SinAngle(inclination=None)  # isotropic prior
    distance_prior = Uniform(distance=(10, 100))
    tc_prior = Uniform(tc=(m.time - 0.1, m.time + 0.1))
    prior = JointDistribution(parameters, inclination_prior, distance_prior, tc_prior)  # Joint prior

    return prior


# Likelihoods
"""
The models starting with `test_` are analytic models. These have predefined likelihood functions
that are given by some standard distributions used in testing samplers. The other models are for GW astronomy:
they take in data and calculate a likelihood using an inner product between the data and a signal model.
Currently, all of the gravitational-wave models in PyCBC assume that the data is stationary Gaussian noise
in the absence of a signal. The difference between the models is they make varying simplfying assumptions,
in order to speed up likelihood evaluation. marginalized_phase, marginalized_polarization,
brute_parallel_gaussian_marginalize, single_template, relative assumes simplying assumptions.
"""


# Using TestNormal model: Analytical models (no data is used). Also it is possible to not provide prior to likelihood
def normal_likelihood(parameters, mean, prior):
    # sourcery skip: inline-immediately-returned-variable
    """
    parameters should be ((tuple of) string(s)) , mean (array-like, optional) Default=0,
    cov (array-like, optional) Default: diag terms =1 (var =1) , non diag =0.
    prior should be of class JOint distribution. All other parameters of likelihood should be in dictionary format
    """

    test_likelihood = models.TestNormal(parameters, mean=mean, prior=prior)
    return test_likelihood


# data should be in frequency and dictionary format (keyed by observatory short name such as 'H1', 'L1', 'V1'),
#  static, low_freq_cutoff and psd should be in dict
def gw_likelihood(parameters, static, data_freq, psds, prior):
    # sourcery skip: inline-immediately-returned-variable
    """
    Using SingleTemplate model; contains the definition of the likelihood function you want to explore and
    details the parameters you are using. This model is useful when we know the intrinsic parameters of a source
    (i.e. component masses, spins), but we don't know the extrinsic parameters (i.e. sky location, distance, binary orientation)
    Fixing intrinsic parameters means that we don't have to generate waveforms at every single likelihood evaluation.
    """
    print("Likelihood calculation")
    lklhood = models.SingleTemplate(
        parameters,
        static_params=static,
        prior=prior,
        data_freq=data_freq,
        psds=psds,
        low_frequency_cutoff={"H1": 25, "L1": 25, "V1": 25},
        sample_rate=8192,
    )
    # Takes data of all the detectors, makes likelihood of all three and then multiply them to give us one likelihood func
    return lklhood


# Samplers
# Using Emcee sampler to run MCMC
def emcee(normal_likelihood, nwalkers, iterations, nprocesses):
    print("Number of nwalkers and iterations are ", nwalkers, iterations)
    engine = sampler.EmceeEnsembleSampler(normal_likelihood, nwalkers, nprocesses, use_mpi=True)
    # engine.set_p0(prior = Uniform(x = (-2,2), y = (-3,3))) # Set intial position of the walkers when prior is not given
    engine.set_p0()
    """
    Note that we do not need to provide anything to `set_p0` to set the initial positions
    if prior is there. By default, the sampler will draw from the prior. But when initial point needs to
    be given it should not be given as x =1, y =2 in the same way mean of likelihood can't be set as x = 0, y =1.
    example: models.TestNormal(('x', 'y'), mean = (x = 0, y = 1)) is wrong...just put (0,1) in mean. Here, in setting initial
    position of the walkers case, we need to put Uniform(x=(-1,1)).
    """
    print("Started running MCMC sampler")
    engine.run_mcmc(iterations)

    return engine


# While Emcee is sufficient for many problems, EmceePT, a parallel tempered version of Emcee is more effective at most GW data analysis problems.
def emceePT(gw_likelihood, temp, nwalkers, iterations, nprocesses):
    print("Number of nwalkers and iterations are ", nwalkers, iterations)
    # There is one additional parameter we need to give to EmcceePT which is the number of temperatures. The output of
    # this sampler will thus be 3-dimensional (temps x walkers x iterations). The 'coldest' temperature (0) will contain our actual results.
    engine = sampler.EmceePTSampler(gw_likelihood, ntemps=temp, nwalkers=nwalkers, nprocesses=nprocesses, use_mpi=True)
    # Number of temeratures to use in the sampler.
    engine.set_p0()  # If we don't set p0, it will use the models prior to draw initial points!
    print("MCMC run started")
    engine.run_mcmc(iterations)
    print("MCMC run done")

    return engine


if __name__ == "__main__":

    # Global variables
    # Parameters of GW likelihood function definitions
    # parameters which will vary
    parameters = ("distance", "inclination", "tc")
    # parameters which will remain constant
    static = {
        "mass1": 1.3757,
        "mass2": 1.3757,
        "f_lower": 25.0,
        "approximant": "TaylorF2",  # only inspiral waveform model, and it is fast
        "polarization": 0,
        "ra": 3.44615914,  # Sky locations
        "dec": -0.40808407,
    }
    event_name = "GW170817"
    m = Merger(event_name)
    ifos = ["H1", "V1", "L1"]  # List of observatories we'll analyze
    # Storing GW data (timeseries) of all detectors in dictionary format
    data = {}
    # Storing GW data (frequency domain)  of all detectors in dictionary format
    data_freq = {}
    psds = {}  # Storing power spectral density  of all detectors in dictionary format
    for i in ifos:
        print("Reading GW timeseries and freq domain data for detector ", i)
        data[i], data_freq[i] = gw_data(detector=i, event_name=event_name)
        print("Estimating PSD for detector ", i)
        psds[i] = PSD(data[i])

    print("Estimating prior")
    prior = prior_func(parameters)

    print("Estimating Likelihood")
    lklhood = gw_likelihood(parameters=parameters, static=static, data_freq=data_freq, psds=psds, prior=prior)
    # data and psd should be in dictionaries for gw likelihood

    # Paramters of MCMC run
    nwalkers = 500  # No. of chains/walkers to be used
    iterations = 300  # No. of iterations to be used
    nprocesses = 4  # no. of processes to be used
    temp = 3  # No. of temperatures to be used

    engine = emceePT(lklhood, temp=temp, nwalkers=nwalkers, iterations=iterations, nprocesses=nprocesses)

    # Collecting parameter values at each iteration of the walker at each temperature.
    par_final = np.zeros((temp, nwalkers, iterations * len(parameters)))
    # iterations * length of parameters coz appending column wise
    for i in range(len(parameters)):
        if i == 0:
            # no. of walkers for each parameters which needs to be varied
            par_final = engine.samples[parameters[i]]
            print(par_final.shape, "when i=0")
        else:
            par = engine.samples[parameters[i]]
            par_final = np.append(par_final, par, axis=2)
    par_final = np.append(par_final, engine.model_stats["loglikelihood"], axis=2)
    # Likelihood Takes data of all the detectors, makes likelihood of all three and then multiply them to give us one likelihood func
    # appending likelihood at each iterations (i.e. no. of iteratiosn columns)
    print("Shape of the Final results are ", par_final.shape)

    # Header of the table
    myheader = "Values of different chains at each iterations. Shape is = nchains * (iterations * no. of parameters).\
         Number of slices equal to temperature. Iterations are arranged columns wise and in order 'distance', 'inclination','tc' "
    print("If the above shape is in 3D, then table will have each slice aranged in 2D.")
    print("Writing table")

    # Writing the table
    with open("chainvalues.txt", "w") as outfile:
        for slice_2d in par_final:
            np.savetxt(outfile, slice_2d, fmt="%16.12e", header=myheader, delimiter=",")
            outfile.write("# New slice\n")
