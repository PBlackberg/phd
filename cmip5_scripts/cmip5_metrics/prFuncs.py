import xarray as xr



def calc_rxday(precip):
    rx1day = precip.resample(time='Y').max(dim='time')
    # rx1day_tMean = rx1day.mean(dim=('time'),keep_attrs=True)
    # rx1day_sMean = rx1day.mean(dim=('lat','lon'),keep_attrs=True)


    precip5day = precip.resample(time='5D').mean(dim='time')
    rx5day = precip5day.resample(time='Y').max(dim='time')

    return rx1day, rx5day



def calc_percentiles(precip):
    pr_95 = precip.quantile(0.95,dim=('lat','lon'),keep_attrs=True)
    pr_95 = xr.DataArray(
        data=pr_95.data,
        dims=['time'],
        coords={'time': precip.time.data},
        attrs={'units':'mm/day'}
        )

    pr_97 = precip.quantile(0.97,dim=('lat','lon'),keep_attrs=True)
    pr_97 = xr.DataArray(
        data=pr_97.data,
        dims=['time'],
        coords={'time': precip.time.data},
        attrs={'units':'mm/day'}
        )

    pr_99 = precip.quantile(0.99,dim=('lat','lon'),keep_attrs=True)
    pr_99 = xr.DataArray(
        data=pr_99.data,
        dims=['time'],
        coords={'time': precip.time.data},
        attrs={'units':'mm/day'}
        )

    pr_999 = precip.quantile(0.999,dim=('lat','lon'),keep_attrs=True)
    pr_999 = xr.DataArray(
        data=pr_999.data,
        dims=['time'],
        coords={'time': precip.time.data},
        attrs={'units':'mm/day'}
        )
        
    return pr_95, pr_97, pr_99, pr_999






