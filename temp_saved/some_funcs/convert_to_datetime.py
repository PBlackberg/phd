import datetime
import cftime

def convert_to_datetime(dates, data):
    ''' Some models have cftime datetime format which isn't very nice for plotting.
        Further, models have different calendars. When converting to datetime objects only standard calendar dates are valid.
        In some models the extra day in leap-years is missing. In some models all months have 30 days.
        Calling this function removes the extra days with invalid datetime days. '''
    if isinstance(dates[0], cftime.datetime):
        dates_new, data_new = [], []
        for i, date in enumerate(dates):
            year, month, day = date.year, date.month, date.day
            if month == 2 and day > 29:                                                     # remove 30th of feb                                                
                continue                                                                
            if not (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) and day > 28:   # remove 29th of feb in non-leap years
                continue
            dates_new.append(datetime.datetime(year, month, day))
            data_new.append(data[i])
        data = xr.DataArray(data_new, coords={'time': dates_new}, dims=['time'])
    else:
        dates_new = pd.to_datetime(dates)
    return dates_new, data

