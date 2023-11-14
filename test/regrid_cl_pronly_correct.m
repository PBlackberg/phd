clear all
close all

model_name = 'UKESM1-0-LL';
var_name = 'cl';

load cl_2000_2014

var3D = cl_2000_2014;

b = ncread('cl_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_200001-201412.nc','b');
orog = ncread('cl_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_200001-201412.nc','orog');
%p0 = ncread('cl_Amon_UKESM1-0-LL_ssp585_r1i1p1f2_gn_205001-209912.nc','p0');
%a = ncread('cl_Amon_UKESM1-0-LL_ssp585_r1i1p1f2_gn_205001-209912.nc','a');
lat = ncread('cl_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_200001-201412.nc','lat');
lon = ncread('cl_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_200001-201412.nc','lon');
lev = ncread('cl_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_200001-201412.nc','lev');

for i = 1:length(lev)
    old_z(:,:,i,:) = (lev(i) + b(i)*orog);
end



%%
new_z = linspace(0,15000,30);
new_var3D_z = NaN(length(lon),length(lat),length(new_z),size(var3D,4));

for i = 1:length(lat)
    for j = 1:length(lon)
       for k = 1:size(var3D,4)
            new_var3D_z(j,i,:,k) = interp1(squeeze(double(old_z(j,i,:))),...
                squeeze(double(var3D(j,i,:,k))),squeeze(double(new_z)),'linear');
       end
    end
end

            
p = 101325*(1-((2.25577e-5).*new_z)).^(5.25588);

new_p = [100000;92500;85000;70000;60000;50000;40000;30000;25000;20000;15000;10000;7000;5000;3000;2000;1000;500;100];
new_var3D_p = NaN(length(lon),length(lat),length(new_p));
for i = 1:length(lat)
    for j = 1:length(lon)
        for k = 1:size(var3D,4)
            new_var3D_p(j,i,:,k) = interp1(squeeze(double(p)),...
                squeeze(double(new_var3D_z(j,i,:,k))),squeeze(double(new_p)),'linear');
        end
    end
end

cl = new_var3D_p;
            
save(['cl_interpp_' model_name '_oldlatlon'],'cl','-v7.3')

