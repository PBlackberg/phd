%% Change in PE example script
% compute tropical mean precipitation efficiency as the time and spatial
% mean of precipitation efficiency, defined as the ratio of precipitation
% (pr) to column-integrated cloud condensed water (clwvi) - units of 1/s

% comparing 15 years from historical and ssp585 runs

%% historical PE calculation

    % read in data
    path = '/Volumes/home/edawson/Analysis/historical 2/BCC-CSM2-MR';
    cd(path);
    pr = ncread('pr_2000-2014.nc', 'pr');
    clwvi = ncread('clwvi_2000-2014.nc', 'clwvi');
    lon = ncread('pr_2000-2014.nc', 'lon');
    lat = ncread('pr_2000-2014.nc', 'lat');

    % interpolate to coarse grid (can be skipped if not mapping)
    old_latitude = linspace(lat(1), lat(end), size(pr,2));
    old_longitude = linspace(lon(1), lon(end), size(pr,1));
    old_longitude = old_longitude';
    new_lat = linspace(lat(1), lat(end), 91);
    new_lon = linspace(lon(1), lon(end), 144);
    new_lon = new_lon'; 
 
    interp_pr_hist = NaN(length(new_lon), length(new_lat), size(pr,3));
        for j = 1:size(pr,3)
            interp_pr_hist(:,:,j) = interp2(old_latitude,old_longitude,...
               squeeze(pr(:,:,j)),new_lat,new_lon,'linear');
        end

   interp_clwvi_hist = NaN(length(new_lon), length(new_lat), size(clwvi,3));
        for j = 1:size(clwvi,3)
            interp_clwvi_hist(:,:,j) = interp2(old_latitude,old_longitude,...
               squeeze(clwvi(:,:,j)),new_lat,new_lon,'linear');
        end
    
        clear pr clwvi lon lat old_latitude old_longitude

     % slice out 30N-30S
     idx_30n = find(abs((new_lat-30)) == min(abs(new_lat-30)));
     idx_30s = find(abs((new_lat+30)) == min(abs(new_lat+30)));
     lats = length(idx_30s:1:idx_30n);
     pr_hist_tropics = interp_pr_hist(:,idx_30s:idx_30n,:);
     clwvi_hist_tropics = interp_clwvi_hist(:,idx_30s:idx_30n,:);

     % at every lon,lat,time find ratio of pr to clwvi
     % change pe values to NaN where pr values are <= 1st percentile or
     % where clwvi values are >= 99th percentile

    pe_hist_store = NaN(144,lats,180);

    pr_prc = prctile(pr_hist_tropics,99, 'all');
    clwvi_prc = prctile(clwvi_hist_tropics,1, 'all');

    for lon=1:144
        for lat=1:lats
            for time=1:180
                if clwvi_hist_tropics(lon,lat,time) <= clwvi_prc
                    clwvi_hist_tropics(lon,lat,time) = NaN;
                else
                    clwvi_hist_tropics(lon,lat,time) = clwvi_hist_tropics(lon,lat,time);
                end
                if pr_hist_tropics(lon,lat,time) >= pr_prc
                    pr_hist_tropics(lon,lat,time) = NaN;
                else
                    pr_hist_tropics(lon,lat,time) = pr_hist_tropics(lon,lat,time);
                end
                pe_hist_store(lon,lat,time) = pr_hist_tropics(lon,lat,time)./clwvi_hist_tropics(lon,lat,time);
            end
        end
    end

    % cosine weight to find tropical average
    coslat = cos(pi*new_lat./180);
    trop_coslat = squeeze(coslat(idx_30s:idx_30n));
    hist_lats = squeeze(nanmean(nanmean(pe_hist_store,3),1));
    for z=1:lats
        weighted(z) = trop_coslat(z).*hist_lats(z);
    end
    pe_hist_final = squeeze(sum(weighted, 'all')./(sum(trop_coslat)));

    % check to make sure that there aren't any anomalously high PE values
    % (i.e. different order of magnitude)
    histogram(pe_hist_store)

    clear pr_hist_tropics clwvi_hist_tropics pr_prc clwvi_prc idx_20n idx_20s interp_clwvi_hist interp_pr_hist new_lat new_lon lat lats lon time weighted

    %% ssp585 PE calculation

    % read in data
    path = '/Volumes/home/edawson/Analysis/ssp585/BCC-CSM2-MR';
    cd(path);
    pr = ncread('pr_2085-2099.nc', 'pr');
    clwvi = ncread('clwvi_2085-2099.nc', 'clwvi');
    lon = ncread('pr_2085-2099.nc', 'lon');
    lat = ncread('pr_2085-2099.nc', 'lat');

    % interpolate to coarse grid (can be skipped if not mapping)
    old_latitude = linspace(lat(1), lat(end), size(pr,2));
    old_longitude = linspace(lon(1), lon(end), size(pr,1));
    old_longitude = old_longitude';
    new_lat = linspace(lat(1), lat(end), 91);
    new_lon = linspace(lon(1), lon(end), 144);
    new_lon = new_lon'; 
 
    interp_pr_ssp = NaN(length(new_lon), length(new_lat), size(pr,3));
        for j = 1:size(pr,3)
            interp_pr_ssp(:,:,j) = interp2(old_latitude,old_longitude,...
               squeeze(pr(:,:,j)),new_lat,new_lon,'linear');
        end

   interp_clwvi_ssp = NaN(length(new_lon), length(new_lat), size(clwvi,3));
        for j = 1:size(clwvi,3)
            interp_clwvi_ssp(:,:,j) = interp2(old_latitude,old_longitude,...
               squeeze(clwvi(:,:,j)),new_lat,new_lon,'linear');
        end
    
        clear pr clwvi lon lat old_latitude old_longitude

     % slice out 30N-30S
     idx_30n = find(abs((new_lat-30)) == min(abs(new_lat-30)));
     idx_30s = find(abs((new_lat+30)) == min(abs(new_lat+30)));
     lats = length(idx_30s:1:idx_30n);
     pr_ssp_tropics = interp_pr_ssp(:,idx_30s:idx_30n,:);
     clwvi_ssp_tropics = interp_clwvi_ssp(:,idx_30s:idx_30n,:);

     % at every lon,lat,time find ratio of pr to clwvi
     % change pe values to NaN where pr values are <= 1st percentile or
     % where clwvi values are >= 99th percentile

    pe_ssp_store = NaN(144,lats,180);

    pr_prc = prctile(pr_ssp_tropics,99, 'all');
    clwvi_prc = prctile(clwvi_ssp_tropics,1, 'all');

    for lon=1:144
        for lat=1:lats
            for time=1:180
                if clwvi_ssp_tropics(lon,lat,time) <= clwvi_prc
                    clwvi_ssp_tropics(lon,lat,time) = NaN;
                else
                    clwvi_ssp_tropics(lon,lat,time) = clwvi_ssp_tropics(lon,lat,time);
                end
                if pr_ssp_tropics(lon,lat,time) >= pr_prc
                    pr_ssp_tropics(lon,lat,time) = NaN;
                else
                    pr_ssp_tropics(lon,lat,time) = pr_ssp_tropics(lon,lat,time);
                end
                pe_ssp_store(lon,lat,time) = pr_ssp_tropics(lon,lat,time)./clwvi_ssp_tropics(lon,lat,time);
            end
        end
    end

    % cosine weight to find tropical average
    coslat = cos(pi*new_lat./180);
    trop_coslat = squeeze(coslat(idx_30s:idx_30n));
    ssp_lats = squeeze(nanmean(nanmean(pe_ssp_store,3),1));
    for z=1:lats
        weighted(z) = trop_coslat(z).*ssp_lats(z);
    end
    pe_ssp_final = squeeze(sum(weighted, 'all')./(sum(trop_coslat)));
    
    % check to make sure that there aren't any anomalously high PE values
    histogram(pe_ssp_store)

    clear pr_ssp_tropics clwvi_ssp_tropics pr_prc clwvi_prc idx_20n idx_20s interp_clwvi_ssp interp_pr_ssp new_lat new_lon lat lats lon time

    % change in tropical mean precipitation efficiency
    delta_pe =  pe_ssp_final - pe_hist_final;



