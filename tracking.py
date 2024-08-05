from datetime import datetime, timedelta

def naive_mslp_track(preds, start_lon, start_lat, start_time, end_time):

    time_step = timedelta(hours=6)
    n_steps = int((end_time - start_time) / time_step)    
    time_steps = [start_time + i * time_step for i in range(n_steps)]

    track = []

    for time in time_steps:

        min_mslp_loc = preds['mean_sea_level_pressure'].sel(time=time).sel(lon=slice(115, 129), lat=slice(-21, -10)).argmin(dim=['lat', 'lon'])

    return track