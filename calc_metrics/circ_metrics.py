

def calc_wapArea(wap500, regime):
    if regime == 'area_ascent':
        areaFrac = ((wap500<0)*1).sum(dim=('lat','lon'))/(wap500.shape[1]*wap500.shape[2])*100
        
    if regime == 'area_descent':
        areaFrac = ((wap500>0)*1).sum(dim=('lat','lon'))/(wap500.shape[1]*wap500.shape[2])*100
        
    areaFrac.attrs['units'] = '%'
    return areaFrac















