    elif var == 'hur_calc':                                                                 # Relative humidity (calculated from tempearture and specific humidity)
        q = mF.load_variable(switch, 'hus', dataset, experiment)                            # unitless (kg/kg)
        ta = mF.load_variable(switch, 'ta', dataset, experiment)                            # degrees Kelvin
        p = ta['plev']                                                                      # Pa 
        r = q / (1 - q)                                                                     # q = r / (1+r)
        e_s = 611.2 * np.exp(17.67*(ta-273.15)/(ta-29.66))                                  # saturation water vapor pressure (also: e_s = 2.53*10^11 * np.exp(-B/T) (both from lecture notes), the Goff-Gratch Equation:  10^(10.79574 - (1731.464 / (T + 233.426)))
        r_s = 0.622 * e_s/(p-e_s)                                                           # from book
        da = (r/r_s)*((1+(r_s/0.622)) / (1+(r/0.622)))*100                                  # relative humidity (from book)


