"""Workflow to extract obs data."""
import atexit
import logging
import os

import xarray as xr
import numpy as np
from dask import config as dskconf
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import xclim
import xscen as xs
from xscen.config import CONFIG

# Load configuration
xs.load_config(
    "paths_obs.yml", "config_obs.yml", verbose=(__name__ == "__main__"), reset=True
)

# get logger
if "logging" in CONFIG:
    logger = logging.getLogger("xscen")

if __name__ == "__main__":
    # set dask  configuration
    daskkws = CONFIG["dask"].get("client", {})
    dskconf.set(**{k: v for k, v in CONFIG["dask"].items() if k != "client"})

    # set xclim config to compute indicators on 3H data
    xclim.set_options(data_validation="log")

    # copy config to the top of the log file
    if "logging" in CONFIG and "file" in CONFIG["logging"]["handlers"]:
        f1 = open(CONFIG["logging"]["handlers"]["file"]["filename"], "a+")
        f2 = open("config_obs.yml")
        f1.write(f2.read())
        f1.close()
        f2.close()

    # set email config
    if "scripting" in CONFIG:
        atexit.register(xs.send_mail_on_exit, subject=CONFIG["scripting"]["subject"])

    # initialize Project Catalog (only do this once, if the file doesn't already exist)
    if not os.path.exists(CONFIG["paths"]["project_catalog"]):
        pcat = xs.ProjectCatalog.create(
            CONFIG["paths"]["project_catalog"],
            project=CONFIG["project"],
        )

    # load project catalog
    pcat = xs.ProjectCatalog(CONFIG["paths"]["project_catalog"])

    # set some recurrent variables
    if CONFIG.get("to_dataset_dict", False):
        tdd = CONFIG["to_dataset_dict"]

    # --- EXTRACT---
    if "extract" in CONFIG["tasks"]:
        # iterate on types to extract (reconstruction, simulation)
        for source_type, type_dict in CONFIG["extract"].items():
            # filter catalog for data that we want
            cat = xs.search_data_catalogs(**type_dict["search_data_catalogs"])

            # iterate over ids from the search
            for ds_id, dc in cat.items():
                # attrs of current iteration that are relevant now
                cur = {
                    "id": ds_id,
                    "xrfreq": "D",
                    "processing_level": "extracted",
                }
                # check if steps was already done
                if not pcat.exists_in_cat(**cur):
                    with (
                        Client(**type_dict["dask"], **daskkws),
                        xs.measure_time(name=f"extract {ds_id}", logger=logger),
                    ):
                        # create dataset from sub-catalog with right domain and periods
                        ds_dict = xs.extract_dataset(
                            catalog=dc,
                            **type_dict["extract_dataset"],
                        )

                        # iterate over the different datasets/frequencies
                        for key_freq, ds in ds_dict.items():
                            if type_dict.get("floor", False):
                                # won't be needing this when data is completely cleaned
                                ds["time"] = ds.time.dt.floor(type_dict["floor"])

                            # drop nans and stack lat/lon in 1d loc (makes code faster)
                            if type_dict.get("stack_drop_nans", False):
                                ds = xs.utils.stack_drop_nans(
                                    ds,
                                    ds[list(ds.data_vars)[0]]
                                    .isel(time=0, drop=True)
                                    .notnull(),
                                )

                            # save to zarr
                            path = CONFIG["paths"]["task"].format(**cur)
                            # try:
                            #     xs.save_to_netcdf(ds=ds, filename=path.replace('zarr', 'nc'),
                            #                       rechunk=type_dict["save"]["rechunk"],
                            #                       # for some reason h5netcdf doesn't write correct files here
                            #                       netcdf_kwargs={'engine': 'netcdf4'})
                            # except:
                            #     print(f'Couln\'t write file: {path.replace("zarr", "nc")}')
                            xs.save_to_zarr(ds=ds, filename=path, **type_dict["save"])
                            pcat.update_from_ds(ds=ds, path=path)

    # --- REGRID ---
    if "regrid" in CONFIG["tasks"]:
        # get input and iter over datasets
        input_dict = pcat.search(**CONFIG["regrid"]["inputs"]).to_dataset_dict(**tdd)
        for key_input, ds_input in input_dict.items():
            cur = {
                "id": ds_input.attrs["cat:id"],
                "xrfreq": ds_input.attrs["cat:xrfreq"],
                "processing_level": "regridded",
            }
            if not pcat.exists_in_cat(**cur):
                with (
                    Client(**CONFIG["regrid"]["dask"], **daskkws),
                    xs.measure_time(name=f"regrid {key_input}", logger=logger),
                ):
                    # get output grid
                    ds_grid = pcat.search(**CONFIG["regrid"]["output"]).to_dataset(
                        **tdd
                    )

                    # do regridding
                    ds_regrid = xs.regrid_dataset(
                        ds=ds_input,
                        ds_grid=ds_grid,
                    )

                # save to zarr
                path = f"{CONFIG['paths']['task']}".format(**cur)
                xs.save_to_zarr(ds=ds_regrid, filename=path, **CONFIG["regrid"]["save"])
                pcat.update_from_ds(ds=ds_regrid, path=path, info_dict=cur)

    # --- CLEAN UP ---
    if "cleanup" in CONFIG["tasks"]:
        # get all datasets to clean up and iter
        cu_cats = xs.search_data_catalogs(**CONFIG["cleanup"]["search_data_catalogs"])
        for cu_id, cu_cat in cu_cats.items():
            cur = {
                "id": cu_id,
                "xrfreq": "D",
                "processing_level": "cleaned",
            }
            if not pcat.exists_in_cat(**cur):
                with (
                    Client(**CONFIG["cleanup"]["dask"], **daskkws),
                    xs.measure_time(name=f"clean {cu_id}", logger=logger),
                ):
                    # put the individually adjusted variables back together in one ds
                    freq_dict = xs.extract_dataset(catalog=cu_cat)

                    # iter over dataset of different frequencies (usually just 'D')
                    for key_freq, ds in freq_dict.items():
                        # clean up the dataset
                        ds_clean = xs.clean_up(
                            ds=ds,
                            **CONFIG["cleanup"]["xscen_clean_up"],
                        )

                        # save and update
                        path = f"{CONFIG['paths']['task']}".format(**cur)
                        xs.save_to_zarr(ds_clean, path, **CONFIG["cleanup"]["save"])
                        pcat.update_from_ds(ds=ds_clean, path=path)

    # --- INDICATORS ---
    if "indicators" in CONFIG["tasks"]:
        # get input and iter
        dict_input = pcat.search(**CONFIG["indicators"]["inputs"]).to_dataset_dict(
            **tdd
        )
        for key_input, ds_input in dict_input.items():
            with (
                Client(**CONFIG["indicators"]["dask"], **daskkws),
                xs.measure_time(name=f"indicators {key_input}", logger=logger),
            ):

                # compute indicators
                dict_indicator = xs.compute_indicators(
                    ds=ds_input,
                    indicators=xs.indicators.select_inds_for_avail_vars(ds_input, CONFIG["indicators"]["path_yml"]),
                )

                # iter over output dictionary (keys are freqs)
                for key_freq, ds_ind in dict_indicator.items():
                    cur = {
                        "id": ds_input.attrs["cat:id"],
                        "xrfreq": key_freq,
                        "processing_level": "indicators",
                    }
                    if not pcat.exists_in_cat(**cur):
                        # save to zarr
                        path_ind = f"{CONFIG['paths']['task']}".format(**cur)
                        xs.save_to_zarr(
                            ds_ind, path_ind, **CONFIG["indicators"]["save"]
                        )
                        pcat.update_from_ds(ds=ds_ind, path=path_ind)

    # --- CLIMATOLOGIES ---
    if "climatologies" in CONFIG["tasks"]:
        # iterate over inputs
        ind_dict = pcat.search(**CONFIG["aggregate"]["input"]["obs"]).to_dataset_dict(
            **tdd
        )
        for key_input, ds_input in sorted(ind_dict.items()):
            cur = {
                "id": ds_input.attrs["cat:id"],
                "xrfreq": ds_input.attrs["cat:xrfreq"],
                "processing_level": "climatology",
            }
            # filter for testing
            # only do ERA5 for now and crop it!
            # if not all(s in key_input for s in ['ERA5', 'MS']):
            #     continue
            # ds_input = ds_input.isel(lat=slice(100, 125), lon=slice(100, 120), )

            if not pcat.exists_in_cat(**cur):
                with (Client(**CONFIG["aggregate"]["dask"], **daskkws) as client,
                      xs.measure_time(name=f"climatologies {key_input}", logger=logger)
                      ):
                    # compute climatological mean
                    all_periods = []
                    for period in CONFIG["aggregate"]["periods"]:
                        # # skip all data except bcs for period 1980-1985 ToDo: remove when tests finished
                        # if int(period[0]) == 1980 and int(period[1]) == 1985 and 'bcs' not in key_input:
                        #     continue
                        # compute properties for period when contained in data
                        if ds_input.time.dt.year.min() <= int(period[0]) and \
                                ds_input.time.dt.year.max() >= int(period[1]):

                            logger.info(f"Computing climatology for {key_input} for period {period}")

                            # Calculate climatological mean --------------------------------
                            logger.info(f"Computing climatological mean for {key_input} for period {period}")
                            ds_mean = xs.aggregate.climatological_op(
                                ds=ds_input,
                                **CONFIG["aggregate"]["climatological_mean"],
                                periods=period,
                                rename_variables=True,
                                periods_as_dim=True,
                            )
                            # ds_mean = ds_mean.assign_coords(period=f'{period[0]}-{period[1]}')
                            # ds_mean = ds_mean.expand_dims(dim='period')
                            # ds_mean = ds_mean.drop_vars('horizon')
                            all_periods.append(ds_mean)

                            # Calculate interannual standard deviation, skipping intra-[freq] std --------------------
                            logger.info(f"Computing interannual standard deviation for {key_input} for period {period}")
                            ds_input_std = ds_input.filter_by_attrs(
                                description=lambda s: 'standard deviation' not in str(s)
                            )
                            ds_std = xs.aggregate.climatological_op(
                                ds=ds_input_std,
                                **CONFIG["aggregate"]["climatological_std"],
                                periods=period,
                                rename_variables=True,
                                periods_as_dim=True,
                            )
                            # ds_std = ds_std.assign_coords(period=f'{period[0]}-{period[1]}')
                            # ds_std = ds_std.expand_dims(dim='period')
                            # ds_std = ds_std.drop_vars('horizon')
                            all_periods.append(ds_std)

                            # Calculate climatological standard deviation --------------------
                            logger.info(
                                f"Computing climatological standard deviation for {key_input} for period {period}")
                            # ToDo: This could be generic to work for all terms involved, depending on the input freq
                            ds_std_clim = xr.Dataset()
                            with xr.set_options(keep_attrs=True):
                                for v_type in set([name.split('_')[0] for name in ds_input.data_vars]):
                                    # pick the interannual standard deviation of the variable
                                    ds_std_varname = [n for n in ds_std.data_vars if v_type in n and 'std' in n]
                                    if len(ds_std_varname) == 1:
                                        ds_std_varname = ds_std_varname[0]
                                    else:
                                        raise ValueError(
                                            f"More than one variable found containing "
                                            f"'{v_type}' and 'std' in {ds_std_varname}"
                                        )
                                    # pick the mean of intra-annual/monthly/seasonal standard deviation of the variable
                                    ds_mean_varname = [n for n in ds_mean.data_vars if v_type in n and 'std' in n]
                                    if len(ds_mean_varname) == 1:
                                        ds_mean_varname = ds_mean_varname[0]
                                    else:
                                        raise ValueError(
                                            f"More than one variable found containing "
                                            f"'{v_type}' and 'std' in {ds_mean_varname}"
                                        )
                                    # calculate the total climatological standard deviation
                                    new_varname = f'{v_type}_std_clim_total'
                                    ds_std_clim[new_varname] = np.sqrt(
                                        np.square(ds_std[ds_std_varname]) +
                                        np.square(ds_mean[ds_mean_varname])
                                    )
                                    # add attributes
                                    ds_std_clim[new_varname].attrs['description'] = \
                                        f"{xclim.core.formatting.default_formatter.format_field(ds_mean.attrs['cat:xrfreq'], 'adj').capitalize()} " \
                                        f"total standard deviation of {' '.join(ds_std[ds_std_varname].attrs['description'].split(' ')[-3:])}"
                                    ds_std_clim[new_varname].attrs['long_name'] = ds_std_clim[new_varname].attrs[
                                        'description']
                            all_periods.append(ds_std_clim)

                            # Calculate trends -----------------------------------------------
                            logger.info(f"Computing climatological linregress for {key_input} for period {period}")

                            # for testing DJF in seasonal change the period - ToDo: remove!
                            # if 'QS-DEC' in key_input:
                            #     period = [str(int(year) - 1) for year in period]

                            ds_input_trend = ds_input[[v for v in ds_input.data_vars if 'mean' in v]]
                            # .isel(lat=slice(100, 105), lon=slice(100, 105), )
                            ds_trend = xs.aggregate.climatological_op(
                                ds=ds_input_trend,
                                **CONFIG["aggregate"]["climatological_trend"],
                                periods=period,
                                min_periods=0.7,
                                rename_variables=True,
                                periods_as_dim=True,
                            )
                            # ds_trend = ds_trend.assign_coords(period=f'{period[0]}-{period[1]}')
                            # ds_trend = ds_trend.expand_dims(dim='period')
                            # ds_trend = ds_trend.drop_vars('horizon')
                            all_periods.append(ds_trend)

                    # # remove all dates so that periods can be merged
                    # new_time = {
                    #     1: {'year': ['ANN']},
                    #     4: {'season': ['MAM', 'JJA', 'SON', 'DJF']},
                    #     12: {'month': list(xr.coding.cftime_offsets._MONTH_ABBREVIATIONS.values())},
                    # }  # calendar.month_abbr[1:]
                    # all_periods = [ds.rename({'time': list(new_time[ds.time.size].keys())[0]})
                    #                .assign_coords(new_time[ds.time.size]) for ds in all_periods]
                    logger.info(f"Merging climatology of periods for {key_input}")
                    #all_periods = client.scatter(all_periods)
                    ds_clim = xr.merge(all_periods, combine_attrs='override')

                    # save to zarr
                    # client.scatter(ds_clim)
                    path = f"{CONFIG['paths']['task']}".format(**cur)
                    xs.save_to_zarr(ds_clim, path, **CONFIG["aggregate"]["save"])
                    pcat.update_from_ds(ds=ds_clim.drop_vars(['time']), path=path)

    # --- PLOTTING ---
    if "plotting" in CONFIG["tasks"]:
        # get input and iter
        dict_input = pcat.search(**CONFIG["plotting"]["input"]["data"]).to_dataset_dict(**tdd)
        for key_input, ds_input in sorted(dict_input.items()):
            with (
                Client(**CONFIG["plotting"]["dask"], **daskkws),
                xs.measure_time(name=f"plotting {key_input}", logger=logger),
            ):
                # practice with monthly or seasonal data
                # if 'AS-JAN' in key_input: continue
                if 'ERA5' not in key_input: continue

                import cartopy.crs as ccrs
                import figanos.matplotlib as fg

                for period in [['haha', 'hihihi']]: #CONFIG["plotting"]["periods"]:
                    for ind in ds_input.data_vars.values():
                        plot_id = f"{ind.attrs['long_name'].lower()} - {ds_input.attrs['source']} ({period})"
                        # logger.info(f"Doing {plot_id}")
                        print(f"{plot_id}")

                        # # use Ouranos style
                        # fg.utils.set_mpl_style('ouranos')
                        # # Selecting a time and slicing our starting Dataset
                        #
                        # # defining our projection.
                        # projection = ccrs.LambertConformal()
                        # ds = ds_input['tg_mean_clim_mean'].isel(period=0, year=0)
                        # fg.gridmap(data, projection=projection,
                        #           features=['coastline', 'ocean'], frame=True, show_time='lower left')


    # --- ENSEMBLES ---
    if "ensembles" in CONFIG["tasks"]:
        # one ensemble (file) per level, per experiment, per xrfreq
        for processing_level in CONFIG["ensembles"]["processing_levels"]:
            ind_df = pcat.search(processing_level=processing_level).df
            # iterate through available xrfreq, exp and variables
            for experiment in ind_df.experiment.unique():
                for xrfreq in ind_df.xrfreq.unique():
                    # get all datasets that go in the ensemble
                    ind_dict = pcat.search(
                        processing_level=processing_level,
                        experiment=experiment,
                        xrfreq=xrfreq,
                    ).to_dataset_dict(**tdd)

                    cur = {
                        "processing_level": f"ensemble-{processing_level}",
                        "experiment": experiment,
                        "xrfreq": xrfreq,
                    }
                    if not pcat.exists_in_cat(**cur):
                        with (
                            Client(**CONFIG["ensembles"]["dask"], **daskkws),
                            xs.measure_time(
                                name=f"ens-{processing_level} {experiment} {xrfreq}",
                                logger=logger,
                            ),
                        ):
                            ens_stats = xs.ensemble_stats(
                                datasets=ind_dict,
                                to_level=f"ensemble-{processing_level}",
                            )

                            # add new id
                            cur["id"] = ens_stats.attrs["cat:id"]

                            # save to zarr
                            path = f"{CONFIG['paths']['task']}".format(**cur)
                            xs.save_to_zarr(
                                # ens_stats, path, **CONFIG["ensembles"]["save"]
                            )
                            pcat.update_from_ds(ds=ens_stats, path=path)

    xs.send_mail(
        subject="ObsFlow - Message",
        msg="Congratulations! All tasks of the workflow were completed!",
    )
