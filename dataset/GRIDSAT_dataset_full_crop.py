# original data
import math
import xarray as xr

from torch.utils.data import Dataset
from petrel_client.client import Client
import csv
import numpy as np
import io
import time

import json
import pandas as pd
import os
import copy
import queue
import torch
import torchvision.transforms as transforms

from datetime import datetime
from datetime import timedelta


from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
import torchvision.transforms.functional as TF
import random




Years = {
    'train': range(2011, 2018),
    'valid': range(2019, 2021),
    'test': range(2018, 2019),
    'all': range(2011, 2021)
}

TCIR_vnames_all = ["data_set", "ID", "lon", "lat", "time", "Vmax", "R35_4qAVG", "MSLP"]

TCIR_inp_vnames = ["IR1", "WV", "VIS", "PMW"]
TCIR_vnames = ["lon", "lat", "time", "Vmax", "R35_4qAVG", "MSLP"]
multi_level_vnames = [
    "z", "t", "q", "r", "u", "v", "vo", "pv",
]
single_level_vnames = [
    "t2m", "u10", "v10", "tcc", "tp", "tisr",
]
long2shortname_dict = {"geopotential": "z", "temperature": "t", "specific_humidity": "q", "relative_humidity": "r", "u_component_of_wind": "u", "v_component_of_wind": "v", "vorticity": "vo", "potential_vorticity": "pv", \
    "2m_temperature": "t2m", "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10", "total_cloud_cover": "tcc", "total_precipitation": "tp", "toa_incident_solar_radiation": "tisr"}
constants = [
    "lsm", "slt", "orography"
]
height_level = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, \
    500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
# height_level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

multi_level_dict_param = {"z":height_level, "t": height_level, "q": height_level, "r": height_level}

from typing import Sequence



class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class GRIDSAT_crop_dataset(Dataset):
    def __init__(self, data_dir='E:\\代码上传\\SETCD_download\\GRIDSAT\\npy_fengwu_era5', split='train', **kwargs) -> None:
        super().__init__()
        #print("init begin")
        self.data_dir = data_dir
        self.save_meanstd_dir = kwargs.get('save_meanstd_dir', 'E:\\MSCAR\\dataset\\mean_std\\npy_fengwu_era5_meanstd_140')

        self.valid_log_name = kwargs.get('valid_log_name', "all_basin")
        self.cfgdir = kwargs.get('cfgdir', "/mnt/petrelfs/wangxinyu1/TC_CNN/VGG")

        self.img_data_nan_rate_threshold = kwargs.get('img_data_nan_rate_threshold', 0.01)

        self.train_begin_year = kwargs.get('train_begin_year', 1980)
        self.train_end_year = kwargs.get('train_end_year', 2017)
        self.valid_begin_year = kwargs.get('valid_begin_year', 2019)
        self.valid_end_year = kwargs.get('valid_end_year', 2020)
        self.test_begin_year = kwargs.get('test_begin_year', 2018)
        self.test_end_year = kwargs.get('test_end_year', 2018)
        Years = {
            'train': range(self.train_begin_year, self.train_end_year+1),
            'valid': range(self.valid_begin_year, self.valid_end_year+1),
            'test' : range(self.test_begin_year,  self.test_end_year +1),
            'all': range(1980, 2021)
        }

        self.all_year = Years[split]
        self.input_length = kwargs.get('input_length', 4)
        self.output_length = kwargs.get('output_length', 4)
        self.output_step_length = kwargs.get('output_step_length', 1)

        self.TCIR_image_size = kwargs.get('TCIR_image_size', 140)
        self.ERA5_image_size = kwargs.get('ERA5_image_size', 40)
        
        self.window_size = self.input_length + self.output_length

        self.resolution = kwargs.get('resolution', 0.25)
        self.GridSat_resolution = kwargs.get('GridSat_resolution', 0.07)
        self.resolution = 1 / self.resolution
        self.GridSat_resolution = 1 / self.GridSat_resolution
        self.radius = kwargs.get('radius', 10)
        self.radius_np = int(self.radius * self.resolution)
        Years_dict = kwargs.get('years', Years)
        self.is_map_inp_intensity = kwargs.get('is_map_inp_intensity', False)

        self.is_save_npy = kwargs.get('is_save_npy', False)
        self.is_load_npy = kwargs.get('is_load_npy', True)
        self.is_use_lifetime_num = kwargs.get("is_use_lifetime_num", False)

        self.inp_type = kwargs.get('inp_type', ["IR", "ERA5", "Seq"])
        self.set_IR_zero = kwargs.get('set_IR_zero', False)
        self.set_ERA5_zero = kwargs.get('set_ERA5_zero', False)
        self.set_Seq_zero = kwargs.get('set_Seq_zero', False)

        vnames_type = kwargs.get("vnames", {})
        self.constants_types = vnames_type.get('constants', [])
        self.ERA5_vnames_dic = self.get_ERA5_dic()
       
        self.single_level_vnames = vnames_type.get('single_level_vnames', ['u10', 'v10', 't2m', 'msl'])
        self.multi_level_vnames = vnames_type.get('multi_level_vnames', ['z', 'q', 'u', 'v', 't'])
        self.height_level_list = vnames_type.get('hight_level_list', [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])

        self.height_level_indexes = [height_level.index(j) for j in self.height_level_list]

        self.label_vnames = vnames_type.get('label_vnames', ["USA_WIND", "USA_PRES"])
        self.TCIR_vnames = vnames_type.get('TCIR_vnames', ["Vmax", "MSLP"])
        # [irwin_cdr, irwvp, vschn]
        self.GRIDSAT_vnames_dic = {"irwin_cdr":0, "irwvp":1, "vschn":2}
        self.GRIDSAT_vnames = vnames_type.get('GRIDSAT_vnames', ["irwin_cdr"])

        
        self.train_label_Basin = vnames_type.get('train_label_Basin', ["EP", "NA", "NI", "SA", "SI", "SP", "WP"])
        self.test_label_Basin = vnames_type.get('test_label_Basin', ["EP", "NA", "NI", "SA", "SI", "SP", "WP"])

            
        self.split = split
        self.data_dir = data_dir
        

        if len(self.constants_types) > 0:
            self.constants_data = self.get_constants_data(self.constants_types)
        else:
            self.constants_data = None
        
        
        self._get_TCIR_meanstd()
        self.intensity_mean, self.intensity_std = self.get_TCIR_label_meanstd()
        # self.GridSat_IR_mean = 265.25488
        # self.GridSat_IR_std = 36.66675
        self.GridSat_IR_mean, self.GridSat_IR_std = self.get_GRIDSAT_IR_meanstd()
        self.era5_mean, self.era5_std = self.get_era5_crop_meanstd()
        # print("TCIR和EAR5均值大小")
        # print(self.intensity_mean, self.intensity_std)
        # print(self.era5_mean.shape, self.era5_std.shape)

        self.data_element_num = len(self.single_level_vnames) + len(self.multi_level_vnames) * len(self.height_level_list)



        self.irwin_cdr_nan_IR, self.irwvp_nan_IR, self.vschn_nan_IR, self.all_day_list = self.nan_IR_data()

        #导入TCIR输入路径
        years = Years_dict[split]
        
        self.input_day_list, self.label_day_list, self.input_intensity_final, self.label_intensity_final, self.input_latlon_final = self.init_file_list(years)


        self.is_data_augmentation = kwargs.get('is_data_augmentation', False)

        if (self.is_data_augmentation) and (self.split=="train"):
            self.GRIDSAT_transform_0 =transforms.Compose([
                transforms.CenterCrop(self.TCIR_image_size),
                MyRotateTransform([0])
                ])
            self.ERA5_transform_0 =transforms.Compose([
                transforms.CenterCrop(self.ERA5_image_size),
                MyRotateTransform([0])
                ])
            self.GRIDSAT_transform_90 =transforms.Compose([
                transforms.CenterCrop(self.TCIR_image_size),
                MyRotateTransform([90])
                ])
            self.ERA5_transform_90 =transforms.Compose([
                transforms.CenterCrop(self.ERA5_image_size),
                MyRotateTransform([90])
                ])
            self.GRIDSAT_transform_180 =transforms.Compose([
                transforms.CenterCrop(self.TCIR_image_size),
                MyRotateTransform([180])
                ])
            self.ERA5_transform_180 =transforms.Compose([
                transforms.CenterCrop(self.ERA5_image_size),
                MyRotateTransform([180])
                ])
            self.GRIDSAT_transform_270 =transforms.Compose([
                transforms.CenterCrop(self.TCIR_image_size),
                MyRotateTransform([270])
                ])
            self.ERA5_transform_270 =transforms.Compose([
                transforms.CenterCrop(self.ERA5_image_size),
                MyRotateTransform([270])
                ])

            self.rotate_angles = len(self.input_day_list)*[0] + len(self.input_day_list)*[90] + len(self.input_day_list)*[180] + len(self.input_day_list)*[270]
            self.input_day_list = self.input_day_list*4
            self.label_day_list = self.label_day_list*4
            self.input_intensity_final = self.input_intensity_final*4
            self.label_intensity_final = self.label_intensity_final*4
            self.input_latlon_final = self.input_latlon_final*4



        
        self.era5_inp_url = []
        self.GridSat_inp_url = []


        for input_window in self.input_day_list:
            urls = []
            urls_GridSat = []
            for day in input_window:
                url = self.url_to_era5(day)
                url_GridSat = self.url_to_GridSat(day)
                urls.append(url)
                urls_GridSat.append(url_GridSat)

            self.era5_inp_url.append(urls)
            self.GridSat_inp_url.append(urls_GridSat)
        self.era5_inp_url = np.array(self.era5_inp_url)
        self.GridSat_inp_url = np.array(self.GridSat_inp_url)

        self.is_rand_rotation = kwargs.get('is_rand_rotation', False)
        if self.is_rand_rotation:
            self.GRIDSAT_transform =transforms.Compose([
                transforms.CenterCrop(self.TCIR_image_size),
                MyRotateTransform([0, 90, 180, 270])
                ])
            self.ERA5_transform =transforms.Compose([
                transforms.CenterCrop(self.ERA5_image_size),
                MyRotateTransform([0, 90, 180, 270])
                ])
        else:
            self.GRIDSAT_transform =transforms.Compose([
                transforms.CenterCrop(self.TCIR_image_size),
                ])
            self.ERA5_transform =transforms.Compose([
                transforms.CenterCrop(self.ERA5_image_size),
                ])




        self.len_file = len(self.input_day_list)
        print("dataset length:{}".format(self.len_file))
        

        lds_config = kwargs.get("lds_config", {})
        self.use_lds = lds_config.get('use_lds', False)
        reweight = lds_config.get('reweight', 'sqrt_inv')
        lds = lds_config.get('lds', False)
        lds_kernel = lds_config.get('lds_kernel', 'gaussian')
        lds_ks = lds_config.get('lds_ks', 5)
        lds_sigma = lds_config.get('lds_sigma', 2)
        min_label = lds_config.get('min_label', [0, 850])
        max_label = lds_config.get('max_label', [200, 1030])
        
        
        if self.use_lds and (self.split=="train"):
            self.weights = np.zeros((self.len_file, self.output_length, len(self.label_vnames)))
            label_np = np.array(self.label_intensity_final)
            for i in range(self.output_length):
                for j in range(len(self.label_vnames)):
                    self.weights[:, i, j] = self._prepare_weights(labels=label_np[:, i, j], reweight=reweight, max_target=max_label[j], min_target=min_label[j], lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma)

        self.is_diff = kwargs.get("is_diff", False)
        

    def get_sid_isotime(self, ir_list, ir_nan_data):
        for i in range(len(ir_nan_data)):
            tc_name = ir_nan_data["SID"].iloc[i]
            tc_ISO_TIME = ir_nan_data["ISO_TIME"].iloc[i]
            ir_list.append(str(tc_name)+"_"+str(tc_ISO_TIME))
        return ir_list

    def nan_IR_data(self):
        irwin_cdr_nan_IR = []
        irwvp_nan_IR = []
        vschn_nan_IR = []
        
        all_day_list = []

        csv_data = pd.read_csv('E:\\MSCAR\\statistics_file\\GRIDSAT_crop_dataset_statistics_true_1980_2022.csv')
        
        irwin_cdr_nan_data = csv_data.loc[(csv_data["irwin_cdr_nan_rate"]>self.img_data_nan_rate_threshold)]
        irwvp_nan_data = csv_data.loc[(csv_data["irwvp_nan_rate"]>self.img_data_nan_rate_threshold)]
        vschn_nan_data = csv_data.loc[(csv_data["vschn_nan_rate"]>self.img_data_nan_rate_threshold)]
        
        irwin_cdr_nan_IR = self.get_sid_isotime(irwin_cdr_nan_IR, irwin_cdr_nan_data)
        irwvp_nan_IR = self.get_sid_isotime(irwvp_nan_IR, irwvp_nan_data)
        vschn_nan_IR = self.get_sid_isotime(vschn_nan_IR, vschn_nan_data)

        for i in range(len(csv_data)):
            tc_name = csv_data["SID"].iloc[i]
            tc_ISO_TIME = csv_data["ISO_TIME"].iloc[i]
            all_day_list.append(str(tc_name)+"_"+str(tc_ISO_TIME))


        return irwin_cdr_nan_IR, irwvp_nan_IR, vschn_nan_IR, all_day_list



    def is_IR_data_useful(self, sid, iso_time):
        name = str(sid)+"_"+str(iso_time)
        flag = True
        if name in self.all_day_list:
            if "irwin_cdr" in self.GRIDSAT_vnames:
                if name in self.irwin_cdr_nan_IR:
                    flag = False
            if "irwvp" in self.GRIDSAT_vnames:
                if name in self.irwvp_nan_IR:
                    flag = False
            if "vschn" in self.GRIDSAT_vnames:
                if name in self.vschn_nan_IR:
                    flag = False
        else:
            flag = False
        return flag

        

    def get_ERA5_dic(self):
        ERA5_vnames_dic = {}
        
        single_level_vnames = ['u10', 'v10', 't2m', 'msl']
        multi_level_vnames = ['z', 'q', 'u', 'v', 't']
        height_level_list = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        index = 0
        for vname in single_level_vnames:
            ERA5_vnames_dic[vname] = index
            index += 1
        for vname in multi_level_vnames:
            ERA5_vnames_dic[vname] = {}
            for height in height_level_list:
                ERA5_vnames_dic[vname][height] = index
                index += 1
        return ERA5_vnames_dic

    def init_file_list(self, years):
        IBTrACS_url = "E:\\MSCAR\\dataset\\IBTrACS\\ibtracs.ALL.list.v04r00.csv"
        
        csv_data = pd.read_csv(IBTrACS_url, encoding="utf-8", low_memory=False, )
        input_final = []
        label_final = []
        input_intensity_final = []
        label_intensity_final = []
        input_latlon_final = []
        lifetime_num_dic = {}
        all_tc_len = 0
        if self.split == "train" or self.split == "test":
            label_Basin = self.train_label_Basin
        else:
            label_Basin = self.test_label_Basin

        for year in years:
            
            year_data = csv_data.loc[(csv_data["SEASON"]==str(year))]
            
            all_year_tc = []

            #print(len(year_data))
            for i in range(len(year_data)):
                tc_name = year_data["SID"].iloc[i]
                tc_Basin = year_data["BASIN"].iloc[i]
                
                if pd.isna(tc_Basin):
                    tc_Basin="NA"
                if (tc_name not in all_year_tc) and (tc_Basin in label_Basin):
                    all_year_tc.append(tc_name)
                # if (tc_Basin not in label_Basin):
                #     print(tc_name, tc_Basin)
            print(f"The number of TC in the {label_Basin} basins in {year}")
            print(len(all_year_tc))
            all_tc_len = all_tc_len + len(all_year_tc)

        
            for tc in all_year_tc:
                if self.is_use_lifetime_num:
                    
                    lifetime_num = -1
                tc_data = year_data.loc[(year_data["SID"]==tc)]
                len_tc = len(tc_data)

                if len_tc>=self.window_size:
                    time_need = []
                    label_need = []
                    for i in range(len_tc):
                        iso_time = tc_data["ISO_TIME"].iloc[i]
                        if iso_time[11:] in ["00:00:00", "06:00:00", "12:00:00", "18:00:00"]:
                            if self.is_use_lifetime_num:
                                lifetime_num = lifetime_num + 1
                            day_data = tc_data.loc[(tc_data["ISO_TIME"]==iso_time)]
                            day_label = []
                            flag = self.is_IR_data_useful(sid=tc, iso_time=iso_time)
                            
                            for vname in self.label_vnames:
                                data_vname = np.array(day_data[vname])[0] 
                                
                                if data_vname != ' ':
                                    data_vname = float(data_vname)
                                    day_label.append(data_vname)
                                else:
                                    flag = False
                                    break
                            if flag:
                                
                                iso_time_path = os.path.join(str(year), tc, iso_time)
                                
                                time_need.append(iso_time_path)
                                label_need.append(day_label)
                                if self.is_use_lifetime_num:
                                    lifetime_num_dic[str(iso_time_path)] = lifetime_num
                    input_start = 0
                    input_end   = 0
                    label_start = 0
                    label_end   = 0
                    for j in range(len_tc-self.window_size):
                        
                        input_start = j
                        input_end   = input_start + self.input_length
                        label_start = input_end
                        label_end   = label_start + self.output_length

                        input_day = time_need[input_start:input_end]
                        label_day = time_need[label_start:label_end]

                        window_day = input_day + label_day
                        if len(window_day)!= self.window_size:
                            break
                        if self.check_window(window_day):
                            input_final.append(input_day)
                            label_final.append(label_day)
                            input_intensity = []
                            input_latlon = []
                            for time_url in input_day:
                                in_time = time_url.split('\\')[-1]
                                inp_data = tc_data.loc[(tc_data["ISO_TIME"]==in_time)]
                                day_label = []
                                for vname in self.label_vnames:
                                    data_vname = np.array(inp_data[vname])[0]
                                    data_vname = float(data_vname)
                                    day_label.append(data_vname)
                                if self.is_use_lifetime_num:
                                    day_label.append(lifetime_num_dic[str(time_url)])
                                input_intensity.append(day_label)
                                

                                lat = float(np.array(inp_data["LAT"])[0])
                                lon = float(np.array(inp_data["LON"])[0])
                                input_latlon.append([lat, lon])

                            label_intensity = []
                            
                            for lab_time in label_day:
                                lab_time = lab_time.split('\\')[-1]
                                lab_data = tc_data.loc[(tc_data["ISO_TIME"]==lab_time)]
                                day_label = []
                                for vname in self.label_vnames:
                                    data_vname = np.array(lab_data[vname])[0]
                                    data_vname = float(data_vname)
                                    day_label.append(data_vname)
                                
                                
                                label_intensity.append(day_label)
                            input_latlon_final.append(input_latlon)
                            input_intensity_final.append(input_intensity)
                            label_intensity_final.append(label_intensity)
                else:
                    print(f"Insufficient length of {year}{tc}")
                    continue

           
            # print(time_latlon)
        
        print(f"All TC num: {all_tc_len}")
        return input_final, label_final, input_intensity_final, label_intensity_final, input_latlon_final


    
    def init_file_list_save_pre(self, years):
        IBTrACS_url = "E:\\MSCAR\\dataset\\IBTrACS\\ibtracs.ALL.list.v04r00.csv"
        
        csv_data = pd.read_csv(IBTrACS_url, encoding="utf-8", low_memory=False, )
        headers = ["year", "SID", "ISO_TIME", "USA_ATCF_ID", "USA_STATUS",]
        #"USA_WIND", "USA_WIND_PRE", "USA_PRES", "USA_PRES_PRE",
        for num in range(self.output_length):
            h = (num+1) * 6
            headers = headers + [f"USA_WIND_{h}h"] + [f"USA_WIND_PRE_{h}h"] + [f"USA_PRES_{h}h"] + [f"USA_PRES_PRE_{h}h"]
        basin = self.valid_log_name.split('.')[0]
        path = os.path.join(self.cfgdir, f"Pre_{basin}_{self.valid_begin_year}_{self.valid_end_year}.csv")
        
        with open(path,'w',encoding='utf8',newline='') as f :
            writer = csv.writer(f)
            writer.writerow(headers)
        input_final = []
        label_final = []
        input_intensity_final = []
        label_intensity_final = []
        input_latlon_final = []
        lifetime_num_dic = {}
        if self.split == "train" or self.split == "test":
            label_Basin = self.train_label_Basin
        else:
            label_Basin = self.test_label_Basin

        for year in years:
            
            year_data = csv_data.loc[(csv_data["SEASON"]==str(year))]
            
            all_year_tc = []

            #print(len(year_data))
            for i in range(len(year_data)):
                tc_name = year_data["SID"].iloc[i]
                tc_Basin = year_data["BASIN"].iloc[i]
                
                if pd.isna(tc_Basin):
                    tc_Basin="NA"
                if (tc_name not in all_year_tc) and (tc_Basin in label_Basin):
                    all_year_tc.append(tc_name)
                
            print(f"The number of TC in the {label_Basin} basins in {year}")
            print(len(all_year_tc))


            
            for tc in all_year_tc:
                if self.is_use_lifetime_num:
                    
                    lifetime_num = -1
                tc_data = year_data.loc[(year_data["SID"]==tc)]
                len_tc = len(tc_data)
                
                if len_tc>=self.window_size:
                    time_need = []
                    label_need = []
                    for i in range(len_tc):
                        iso_time = tc_data["ISO_TIME"].iloc[i]
                        if iso_time[11:] in ["00:00:00", "06:00:00", "12:00:00", "18:00:00"]:
                            if self.is_use_lifetime_num:
                                lifetime_num = lifetime_num + 1                            
                            day_data = tc_data.loc[(tc_data["ISO_TIME"]==iso_time)]
                            day_label = []
                            flag = self.is_IR_data_useful(sid=tc, iso_time=iso_time)
                            
                            for vname in self.label_vnames:
                                data_vname = np.array(day_data[vname])[0] 
                                #print(data_vname)
                                if data_vname != ' ':
                                    data_vname = float(data_vname)
                                    day_label.append(data_vname)
                                else:
                                    flag = False
                                    break
                            if flag:
                                #print(day_label)
                                iso_time_path = os.path.join(str(year), tc, iso_time)
                                
                                time_need.append(iso_time_path)
                                label_need.append(day_label)
                                if self.is_use_lifetime_num:
                                    lifetime_num_dic[str(iso_time_path)] = lifetime_num                                
                    #print(time_need)
                    input_start = 0
                    input_end   = 0
                    label_start = 0
                    label_end   = 0
                    for j in range(len_tc-self.window_size):
                        
                        input_start = j
                        input_end   = input_start + self.input_length
                        label_start = input_end
                        label_end   = label_start + self.output_length

                        input_day = time_need[input_start:input_end]
                        label_day = time_need[label_start:label_end]

                        window_day = input_day + label_day
                        if len(window_day)!= self.window_size:
                            break
                        if self.check_window(window_day):
                            input_final.append(input_day)
                            label_final.append(label_day)
                            input_intensity = []
                            input_latlon = []
                            for time_url in input_day:
                                in_time = time_url.split('\\')[-1]
                                inp_data = tc_data.loc[(tc_data["ISO_TIME"]==in_time)]
                                day_label = []
                                for vname in self.label_vnames:
                                    data_vname = np.array(inp_data[vname])[0]
                                    data_vname = float(data_vname)
                                    day_label.append(data_vname)
                                if self.is_use_lifetime_num:
                                    day_label.append(lifetime_num_dic[str(time_url)])                                    
                                input_intensity.append(day_label)

                                lat = float(np.array(inp_data["LAT"])[0])
                                lon = float(np.array(inp_data["LON"])[0])
                                input_latlon.append([lat, lon])

                            label_intensity = []
                            
                            for lab_time in label_day:
                                lab_time = lab_time.split('\\')[-1]
                                lab_data = tc_data.loc[(tc_data["ISO_TIME"]==lab_time)]
                                day_label = []
                                for vname in self.label_vnames:
                                    data_vname = np.array(lab_data[vname])[0]
                                    data_vname = float(data_vname)
                                    day_label.append(data_vname)
                                
                                
                                label_intensity.append(day_label)

                            last_day = input_day[-1].split('\\')[-1]
                            last_inp_day = tc_data.loc[(tc_data["ISO_TIME"]==last_day)]
            
                            rows = {"year":year, "SID":tc, "ISO_TIME":last_day, "USA_ATCF_ID":np.array(last_inp_day["USA_ATCF_ID"])[0], "USA_STATUS":np.array(last_inp_day["USA_STATUS"])[0],}
                            #"USA_WIND", "USA_WIND_PRE", "USA_PRES", "USA_PRES_PRE",
                            for num in range(self.output_length):
                                h = (num+1) * 6
                                num_iso = label_day[num].split('\\')[-1]
                                num_day = tc_data.loc[(tc_data["ISO_TIME"]==num_iso)]
                                rows[f"USA_WIND_{h}h"] = np.array(num_day["USA_WIND"])[0]
                                rows[f"USA_WIND_PRE_{h}h"] = " "
                                rows[f"USA_PRES_{h}h"] = np.array(num_day["USA_PRES"])[0]
                                rows[f"USA_PRES_PRE_{h}h"] = " "


                            basin = self.valid_log_name.split('.')[0]
                            path = os.path.join(self.cfgdir, f"Pre_{basin}_{self.valid_begin_year}_{self.valid_end_year}.csv")
                            #with open(f'/mnt/petrelfs/wangxinyu1/TC_CNN/VGG/GRIDSAT_crop_dataset_pre_{years[0]}_{years[-1]}.csv','a',encoding='utf8',newline='') as f :
                            with open(path,'a',encoding='utf8',newline='') as f :
                                writer = csv.DictWriter(f, fieldnames=rows.keys())
                                writer.writerow(rows)


                            input_latlon_final.append(input_latlon)
                            input_intensity_final.append(input_intensity)
                            label_intensity_final.append(label_intensity)
                
                else:
                    print(f"Insufficient length of {year}{tc}")
                    continue


            # print(time_latlon)
        return input_final, label_final, input_intensity_final, label_intensity_final, input_latlon_final


      
    def is_order(self, day1, day2):
        day1 = day1.split('\\')[-1]
        day2 = day2.split('\\')[-1]
        day1 = datetime(int(day1[0:4]), int(day1[5:7]), int(day1[8:10]), int(day1[11:13]), 0, 0)
        day2 = datetime(int(day2[0:4]), int(day2[5:7]), int(day2[8:10]), int(day2[11:13]), 0, 0)
        hour = day2 - day1
        
        if hour.__str__() == "6:00:00":
            return True
        else:
            return False
        
    def check_window(self, day_list):
        
        for i in range(len(day_list)-1):
            if self.is_order(day_list[i], day_list[i+1]):
                continue
            else:
                return False
        return True
                


    def get_era5_meanstd(self):
        return_data_mean = []
        return_data_std = []
        
        for vname in self.single_level_vnames:
            return_data_mean.append(self.mean_std["mean"][vname])
            return_data_std.append(self.mean_std["std"][vname])
        for vname in self.multi_level_vnames:
            return_data_mean.append(self.mean_std["mean"][vname][self.height_level_indexes])
            return_data_std.append(self.mean_std["std"][vname][self.height_level_indexes])
        return torch.from_numpy(np.concatenate(return_data_mean, axis=0)[:, 0, 0]), torch.from_numpy(np.concatenate(return_data_std, axis=0)[:, 0, 0])

    def get_era5_crop_meanstd(self):
        
        with open(f'{self.save_meanstd_dir}//ERA5_single_TC_mean_std.json',mode='r') as f:
            era5_crop_single_mean_std = json.load(f)
        with open(f'{self.save_meanstd_dir}//ERA5_TC_mean_std.json',mode='r') as f:
            era5_crop_mutil_mean_std = json.load(f)
        era5_crop_mean = []
        era5_crop_std = []
        
        for vname in self.single_level_vnames:
            vname = str(vname)
            era5_crop_mean.append(np.array([era5_crop_single_mean_std["mean"][vname]]))
            era5_crop_std.append(np.array([era5_crop_single_mean_std["std"][vname]]))
        for vname in self.multi_level_vnames:
            for height in self.height_level_list:
                vname = str(vname)
                height = str(height)
                
                era5_crop_mean.append(np.array([era5_crop_mutil_mean_std["mean"][vname][height]]))
                era5_crop_std.append(np.array([era5_crop_mutil_mean_std["std"][vname][height]]))

        return torch.from_numpy(np.concatenate(era5_crop_mean, axis=0)), torch.from_numpy(np.concatenate(era5_crop_std, axis=0))
    


    def _get_TCIR_meanstd(self):
        with open('E:\\MSCAR\\dataset\\mean_std\\TCIR_mean_std.json',mode='r') as f:
            TCIR_mean_std = json.load(f)
    
        self.TCIR_mean_std = {}
        
        self.TCIR_mean_std = TCIR_mean_std
        
        self.TCIR_mean_output_std = {"mean":{}, "std":{}}
        self.TCIR_mean_input_std = {"mean":{}, "std":{}}
        for vname in self.TCIR_vnames:
            index = TCIR_vnames.index(vname)
            #print(self.TCIR_mean_std["label"]["mean"][index])
            self.TCIR_mean_output_std["mean"][vname] = np.array(self.TCIR_mean_std["label"]["mean"][index])
            self.TCIR_mean_output_std["std"][vname] = np.array(self.TCIR_mean_std["label"]["std"][index])
        
        #print(self.TCIR_mean_std)
        
    def get_TCIR_label_meanstd(self):
        out_data_mean = []
        out_data_std = []
        
        for vname in self.TCIR_vnames:
            out_data_mean.append(self.TCIR_mean_output_std["mean"][vname])
            out_data_std.append(self.TCIR_mean_output_std["std"][vname])
        
        return torch.from_numpy(np.array(out_data_mean)), torch.from_numpy(np.array(out_data_std))
    


    
    def get_GRIDSAT_IR_meanstd(self):
        
        with open('E:\\MSCAR\dataset\\mean_std\\npy_fengwu_era5_meanstd_140\\GRIDSAT_TC_mean_std.json',mode='r') as f:
            GRIDSAT_IR_mean_std = json.load(f)
              
        GRIDSAT_IR_mean = []
        GRIDSAT_IR_std = []
        for vname in self.GRIDSAT_vnames:
            
            #print(self.TCIR_mean_std["label"]["mean"][index])
            GRIDSAT_IR_mean.append(np.array([GRIDSAT_IR_mean_std["mean"][vname]]))
            GRIDSAT_IR_std.append(np.array([GRIDSAT_IR_mean_std["std"][vname]]))
        return torch.from_numpy(np.concatenate(GRIDSAT_IR_mean, axis=0)), torch.from_numpy(np.concatenate(GRIDSAT_IR_std, axis=0))
    
    
    
    def get_meanstd(self):
        return self.intensity_mean, self.intensity_std
 
    def url_to_era5(self, url):
        
        aim = url.split('\\')[-1]
        # #print(aim)
        year = aim[0:4]
        month = aim[5:7]
        day = aim[8:10]
        hour = aim[11:13]
        y_m_d = year + '-' + month + '-' + day
        h_m_s = hour + ':' + "00" + ':' + "00"
        era5_url = []
        era5_url_base = os.path.join(url, h_m_s)
        # era5_url_base = f"{self.data_dir}/{era5_url_base}"
        for vname in self.single_level_vnames:
            url_vname = f"{era5_url_base}-{vname}.npy"
            url_vname = f"{self.data_dir}\{url_vname}"
            era5_url.append(url_vname)
        for vname in self.multi_level_vnames:
            for height in self.height_level_list:
                url_vname = f"{era5_url_base}-{vname}-{height}.0.npy"
                url_vname = f"{self.data_dir}\{url_vname}"
                era5_url.append(url_vname)
        return era5_url
    

    def url_to_GridSat(self, url):
        
        
        GRIDSAT_url = f"{self.data_dir}/{url}/GRIDSAT_data.npy"
        
        return GRIDSAT_url

    def get_data(self, url):
                
        try:
            data = np.load(url)
        except Exception as err:
            raise ValueError(f"{url}")
        return data



    def test_pic(self, name, data, cmap='gist_rainbow'):
        import matplotlib.pyplot as plt
        plt.imshow(data, cmap=cmap)
        plt.axis('off')
        plt.savefig(name, transparent=True)  


    def load_era5_full(self, index):
        era5_inp_urls = self.era5_inp_url[index]
        #print(era5_inp_urls)
        era5_inp_data = []
        
        
        for i in range(era5_inp_urls.shape[0]):
            era5_inp_v = []
            
            full_url_list = era5_inp_urls[i, 0].split("\\")[:-1]
            full_url_list[-1] = full_url_list[-1].replace(':', '_')
            full_url_list.append("ERA5_data.npy")
            full_url = ""
            for i in range(len(full_url_list)):
                full_url = full_url + full_url_list[i]
                if i != (len(full_url_list)-1):
                    full_url = full_url + "//"
            
            
            #print(full_url)
            full_data = self.get_data(full_url)
            index_list = []

            for vname in self.single_level_vnames:
                index_list.append(self.ERA5_vnames_dic[vname])
                
            for vname in self.multi_level_vnames:
                for height in self.height_level_list:
                    
                    index_list.append(self.ERA5_vnames_dic[vname][height])

            era5_inp_v = full_data[index_list, :, :]
            #print(f"ERA5 index_list: {index_list}")
            era5_inp_data.append(era5_inp_v)
        era5_inp_data = np.array(era5_inp_data)
  
        return era5_inp_data




    def load_era5(self, index):
        era5_inp_urls = self.era5_inp_url[index]
        #print(era5_inp_urls)
        era5_inp_data = []
        for i in range(era5_inp_urls.shape[0]):
            era5_inp_v = []
            for j in range(era5_inp_urls.shape[1]):
                
                data = self.get_data(era5_inp_urls[i, j])
                
                era5_inp_v.append(data)
            era5_inp_data.append(era5_inp_v)
        era5_inp_data = np.array(era5_inp_data)
        
        return era5_inp_data
    
    def load_GRIDSAT(self, index):
        GridSat_inp_urls = self.GridSat_inp_url[index]
  
        GridSat_inp_urls = np.array(GridSat_inp_urls)
        #print(era5_inp_urls[0,0])
        GridSat_inp_data = []
        #按时序查询
        for i in range(GridSat_inp_urls.shape[0]):
            url = GridSat_inp_urls[i]
            url = url.replace('/', '\\')
            url_list = url.split("\\")
            url_list[-2] = url_list[-2].replace(':', '_')
            url_windows = ""
            for i in range(len(url_list)):
                url_windows = url_windows + url_list[i]
                if i != (len(url_list)-1):
                    url_windows = url_windows + "//"
            url = url_windows
            
            data = self.get_data(url)
            index_list = []
            for vnames in self.GRIDSAT_vnames:
                index_list.append(self.GRIDSAT_vnames_dic[vnames])
            data = data[index_list, :, :]
            #print(f"GRIDAST index_list: {index_list}")
            GridSat_inp_data.append(data)
        GridSat_inp_data = np.array(GridSat_inp_data)
        
        # print("era5读取的大小")
        # print(era5_inp_data.shape)
        return GridSat_inp_data


    def get_IR(self, index, inp_latlon, inp_lable):
        # try:
        #     GridSat_inp = self.load_GRIDSAT(index=index)
        #     #print(GridSat_inp.shape)
        # except:
        #     print(f"{self.GridSat_inp_url[index]} does not exist")
        #     return None
        GridSat_inp = self.load_GRIDSAT(index=index)
        new_GridSat_data = GridSat_inp
        
        new_GridSat_data = np.array(new_GridSat_data)
        new_GridSat_data = torch.from_numpy(new_GridSat_data)
        new_GridSat_data = (new_GridSat_data - self.GridSat_IR_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / self.GridSat_IR_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        new_GridSat_data = torch.where(torch.isnan(new_GridSat_data), torch.full_like(new_GridSat_data, 0), new_GridSat_data)
        
        if (self.is_data_augmentation) and (self.split=="train"):
            angles = self.rotate_angles[index]
            if angles==0:
                new_GridSat_data = self.GRIDSAT_transform_0(new_GridSat_data)
            elif angles==90:
                new_GridSat_data = self.GRIDSAT_transform_90(new_GridSat_data)
            elif angles==180:
                new_GridSat_data = self.GRIDSAT_transform_180(new_GridSat_data)
            elif angles==270:
                new_GridSat_data = self.GRIDSAT_transform_270(new_GridSat_data)
        else:
            new_GridSat_data = self.GRIDSAT_transform(new_GridSat_data)


        if self.is_map_inp_intensity:
            map_GridSat = torch.zeros((new_GridSat_data.shape[0], inp_lable.shape[1], new_GridSat_data.shape[2], new_GridSat_data.shape[3]))
            for t in range(inp_lable.shape[0]):
                for v in range(inp_lable.shape[1]):
                    map_GridSat[t, v, :, :] = inp_lable[t, v]
            new_GridSat_data = torch.concat((new_GridSat_data, map_GridSat), dim=1)
        return new_GridSat_data

    def get_ERA5(self, index, inp_latlon, inp_lable): 

        era5_inp = self.load_era5_full(index=index)
        new_era5_data = era5_inp
        
        new_era5_data = np.array(new_era5_data)
        new_era5_data = torch.from_numpy(new_era5_data)
        new_era5_data = (new_era5_data - self.era5_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / self.era5_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        new_era5_data = self.ERA5_transform(new_era5_data)



        if (self.is_data_augmentation) and (self.split=="train"):
            angles = self.rotate_angles[index]
            if angles==0:
                new_era5_data = self.ERA5_transform_0(new_era5_data)
            elif angles==90:
                new_era5_data = self.ERA5_transform_90(new_era5_data)
            elif angles==180:
                new_era5_data = self.ERA5_transform_180(new_era5_data)
            elif angles==270:
                new_era5_data = self.ERA5_transform_270(new_era5_data)
        else:
            new_era5_data = self.ERA5_transform(new_era5_data)

        
        if self.is_map_inp_intensity:
            map_era5 = torch.zeros((new_era5_data.shape[0], inp_lable.shape[1], new_era5_data.shape[2], new_era5_data.shape[3]))
            for t in range(inp_lable.shape[0]):
                for v in range(inp_lable.shape[1]):
                    map_era5[t, v, :, :] = inp_lable[t, v]
            new_era5_data = torch.concat((new_era5_data, map_era5), dim=1)
            


        return new_era5_data


    def get_lds_kernel_window(self, kernel, ks, sigma):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks)
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

        return kernel_window


    def _prepare_weights(self, labels, reweight, min_target=0, max_target=200, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        labels = labels - min_target
        max_target = max_target - min_target

        value_dict = {x: 0 for x in range(max_target)}
        # labels = self.label_intensity_final
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = self.get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights


    def __getitem__(self, index):
       

        label = self.label_intensity_final[index]
        inp_lable = self.input_intensity_final[index]
        inp_latlon = self.input_latlon_final[index]

        label = np.array(label)
        inp_lable = np.array(inp_lable)
        inp_latlon = np.array(inp_latlon)
        

        label = torch.from_numpy(label) 
        inp_lable = torch.from_numpy(inp_lable)
        # print(label)
        # print(inp_lable)
        label[:, 0] = label[:, 0]
        inp_lable[:, 0] = inp_lable[:, 0]
        

        label = (label - self.intensity_mean.unsqueeze(0)) / self.intensity_std.unsqueeze(0)
        if self.is_use_lifetime_num:
            inp_lable[:, :-1] = (inp_lable[:, :-1] - self.intensity_mean.unsqueeze(0)) / self.intensity_std.unsqueeze(0)
        else:
            inp_lable = (inp_lable - self.intensity_mean.unsqueeze(0)) / self.intensity_std.unsqueeze(0)
        if self.is_diff:
            
            for i in range(label.shape[0]-1, 0, -1):
                label[i] = label[i] - label[i-1]
            label[0] = label[0] - inp_lable[-1]

        label = label[(self.output_step_length-1)::self.output_step_length, :]

        new_GridSat_data = torch.tensor(0)
        new_era5_data = torch.tensor(0)
        #base_dir = "/mnt/petrelfs/wangxinyu1/TC_CNN/VGG/dataset/data_demo_fig"
        if "IR" in self.inp_type:
            new_GridSat_data = self.get_IR(index, inp_latlon, inp_lable)
            if self.set_IR_zero:
                new_GridSat_data = new_GridSat_data * torch.tensor(0)
            # print(self.GridSat_inp_url[index])
            #self.test_pic(f"IR_fig{index}.png", new_GridSat_data[0,0],)
            # self.test_pic(f"{base_dir}/IR_fig1.png", new_GridSat_data[0,0], "Greys")
            # self.test_pic(f"{base_dir}/IR_fig2.png", new_GridSat_data[1,0], "Greys")
            # self.test_pic(f"{base_dir}/IR_fig3.png", new_GridSat_data[2,0], "Greys")
            # self.test_pic(f"{base_dir}/IR_fig4.png", new_GridSat_data[3,0], "Greys")
            # self.test_pic("IR_fig2.png", new_GridSat_data[0,2], "Greys")
            # self.test_pic("IR_fig0.png", new_GridSat_data[0,0], )
            # self.test_pic("IR_fig1.png", new_GridSat_data[0,1], )
            # self.test_pic("IR_fig2.png", new_GridSat_data[0,2], )
            # print(new_GridSat_data[0])
        if "ERA5" in self.inp_type:
            new_era5_data = self.get_ERA5(index, inp_latlon, inp_lable)
            if self.set_ERA5_zero:
                new_era5_data = new_era5_data * torch.tensor(0)
            # for i in range(self.input_length):
            #     self.test_pic(f"{base_dir}/u10_{i}.png", new_era5_data[i,0])
            #     self.test_pic(f"{base_dir}/v10_{i}.png", new_era5_data[i,1])
            #     self.test_pic(f"{base_dir}/t2m_{i}.png", new_era5_data[i,2])
            #     self.test_pic(f"{base_dir}/msl_{i}.png", new_era5_data[i,3])
                                
            # self.test_pic("msl.png", new_era5_data[0,-1])
            # self.test_pic("v850.png", new_era5_data[0,-2])
            # self.test_pic("v750.png", new_era5_data[0,-3])
            # self.test_pic("v700.png", new_era5_data[0,-4])
            # self.test_pic("v550.png", new_era5_data[0,-5])
            # self.test_pic("v500.png", new_era5_data[0,-6])
            # self.test_pic("v350.png", new_era5_data[0,-7])
            # self.test_pic("v300.png", new_era5_data[0,-8])
            # self.test_pic("v200.png", new_era5_data[0,-9])
            # self.test_pic("v150.png", new_era5_data[0,-10])
            # self.test_pic("v50.png", new_era5_data[0,-11])
        if self.set_Seq_zero:
            inp_lable = inp_lable * torch.tensor(0)
        # if self.split=="valid":
        #     input_day = self.input_day_list[index][-1]
        #     year = input_day.split('\\')[0]
        #     sid = input_day.split('\\')[1]
        #     iso_time = input_day.split('\\')[2]
        #     inp_last_tc_day_info = {"ALL_Year":self.all_year,"Year":year, "SID":sid, "ISO_TIME":iso_time}
        #     #print(inp_last_tc_day_info)
        #     return new_GridSat_data, new_era5_data, inp_lable, label, inp_last_tc_day_info
        
        if self.use_lds and self.split=="train":
            weight = torch.tensor(self.weights[index])
            return new_GridSat_data, new_era5_data, inp_lable, label, weight
        
        return new_GridSat_data, new_era5_data, inp_lable, label
    
    def __len__(self):    
        return self.len_file




    
if __name__ == "__main__":

    years = {
        'valid': range(2004, 2005),
        'train': range(2011, 2012),
    }
    #url = os.path.join(dir, )
    data_set = GRIDSAT_crop_dataset(data_dir="E:\\代码上传\\SETCD_download\\GRIDSAT\\npy_fengwu_era5", \
                                    save_meanstd_dir='E:\\MSCAR\\dataset\\mean_std\\npy_fengwu_era5_meanstd_140' ,\
                                    split='train', years=years, inp_type=["IR", "ERA5", "Seq"], is_map_inp_intensity=False,
                            )#vnames={'single_level_vnames':['u10', 'v10', 'msl', 't2m'], 'multi_level_vnames':['z'], 'hight_level_list':[500]}
   
    for i in range(99, 100):
        # begin = time.time()
        # new_GridSat_data, new_era5_data, inp_lable, label, weight = data_set.__getitem__(i)
        # print(new_GridSat_data.shape, new_era5_data.shape, inp_lable.shape, label.shape, weight.shape)
        new_GridSat_data, new_era5_data, inp_lable, label,  = data_set.__getitem__(i)
        print(new_GridSat_data.shape, new_era5_data.shape, inp_lable.shape, label.shape, )
        # print(label)
        print("***************************************new_GridSat_data")
        print(new_GridSat_data,)
        print("***************************************new_era5_data")
        print(new_era5_data,)
        print("***************************************inp_lable")
        print(inp_lable, label,)
        # end = time.time()
        # print("总读取耗时：{}".format(end - begin))


    

