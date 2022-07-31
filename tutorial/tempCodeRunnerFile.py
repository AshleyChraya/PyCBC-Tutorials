 # print("Estimating Likelihood")
    # lklhood = gw_likelihood(parameters = parameters, static = static, data_freq = data_freq, psds  = psds, prior = prior)
    # # data and psd should be in dictionaries for gw likelihood

    # # Paramters of MCMC run
    # nwalkers = 500 # No. of chains/walkers to be used
    # iterations = 300 # No. of iterations to be used 
    # nprocesses = 4 # no. of processes to be used
    # temp = 3 # No. of temperatures to be used

    # engine = emceePT(lklhood,temp = temp, nwalkers = nwalkers, iterations = iterations, nprocesses = nprocesses)
    # #%---------------------------------------------#

    # # Collecting parameter values at each iteration of the walker at each temperature.
    # par_final = np.zeros((temp, nwalkers, iterations*len(parameters))) 
    # # iterations * length of parameters coz appending column wise
    # for i in range(len(parameters)):
    #     if i == 0 :
    #         par_final = engine.samples[parameters[i]]
    #         print(par_final.shape, "when i=0")
    #     else:
    #         par = engine.samples[parameters[i]]
    #         par_final = np.append(par_final, par, axis = 2)
    # par_final = np.append(par_final, engine.model_stats['loglikelihood'], axis = 2) # appending likelihood at each iterations (i.e. no. of iteratiosn columns)
    # print("Shape of the Final results are ", par_final.shape)
    # myheader = "Values of different chains at each iterations. Shape is = nchains * (iterations * no. of parameters).\
    #      Number of slices equal to temperature. Iterations are arranged columns wise and in order \'distance', 'inclination','tc' "
    # print("If the above shape is in 3D, then table will have each slice aranged in 2D.")
    # print("Writing table")
    # # Writing the table
    # with open("chainvalues.txt", 'w') as outfile:
    #     for slice_2d in par_final:
    #         np.savetxt(outfile, slice_2d, fmt = '%16.12e', header = myheader, delimiter = ',')
    #         outfile.write('# New slice\n')