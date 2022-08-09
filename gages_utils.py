import pandas as pd
import numpy as np
import os.path
import re

DIR = 'datasets/GAGESII'

FILENAMES = {
'conterm_basinid.txt':{"include":['STAID','DRAIN_SQKM','HUC02','LAT_GAGE','LNG_GAGE','STATE'],"exclude":[]},
'conterm_bas_classif.txt':{"include":['STAID','CLASS','HYDRO_DISTURB_INDX'],"exclude":[]},
'conterm_bas_morph.txt':{"include":['STAID','BAS_COMPACTNESS'],"exclude":[]},
#'conterm_bound_qa.txt':['STAID',],
#'conterm_climate_ppt_annual.txt':['STAID',],
#'conterm_climate_tmp_annual.txt':['STAID',],
#'conterm_climate.txt':{"include":[], "exclude":[r"\w{3}_\w{3}7100\w+"]}, #Mo
'conterm_climate.txt':{"include":[r".+"],"exclude":[]},
#'conterm_flowrec.txt':['STAID',],
#'conterm_geology.txt':{"include":['STAID','GEOL_REEDBUSH_DOM', 'GEOL_REEDBUSH_DOM_PCT', 'GEOL_HUNT_DOM_CODE', 'GEOL_HUNT_DOM_PCT'], "exclude":[]}, # NEEDS WORK
'conterm_hydromod_dams.txt':{"include":[],"exclude":[r"pre[0-9]{4}_\w+"]}, #Mo
'conterm_hydromod_other.txt':{"include":[r".+"],"exclude":[]}, #Mo
#'conterm_hydro.txt':{"include":[],"exclude":["REACHCODE", r"WB5100_\w{3}_MM"]}, #Mo
'conterm_hydro.txt':{"include":[],"exclude":["REACHCODE"]}, #Mo
'conterm_landscape_pat.txt':{"include":[r".+"],"exclude":[]}, #Mo
'conterm_lc06_basin.txt':{"include":[r".+"],"exclude":[]}, #Mo
'conterm_lc06_mains100.txt':{"include":[r".+"],"exclude":[]},
'conterm_lc06_mains800.txt':{"include":[r".+"],"exclude":[]},
'conterm_lc06_rip100.txt':{"include":[r".+"],"exclude":[]},
'conterm_lc06_rip800.txt':{"include":[r".+"],"exclude":[]},
'conterm_lc_crops.txt':{"include":[r".+"],"exclude":[]},
'conterm_nutrient_app.txt':{"include":[r".+"],"exclude":[]}, #Mo
'conterm_pest_app.txt':{"include":[r".+"],"exclude":[]}, #Mo
'conterm_pop_infrastr.txt':{"include":[r".+"],"exclude":[]},
'conterm_prot_areas.txt':{"include":[r".+"],"exclude":[]},
#'conterm_regions.txt':['STAID',],
'conterm_soils.txt':{"include":[r".+"],"exclude":[]},
'conterm_topo.txt':{"include":[r".+"],"exclude":[]}, #Mo
#'conterm_x_region_names.txt':['STAID',],
}

def first_filter(df, cols_to_keep):

    cols_to_keep_ = []
    for col in df.columns.values:
        #print(col)
        include = 0
        for r in cols_to_keep["include"]:
            #print(re.search(r,col))
            if re.search(r,col):
                include = 1
                break
        #print(include)
        if include == 1:
            cols_to_keep_.append(col)
    if len(cols_to_keep["exclude"]) != 0:
        for col in df.columns.values:
            exclude = 0
            for r in cols_to_keep["exclude"]:
                if re.search(r,col):
                    exclude = 1
                    break
            if exclude == 0:
                cols_to_keep_.append(col)
    
    return cols_to_keep_

def process_raw_columns(df):
    raw_columns = [x for x in df.columns if x.startswith("RAW")]
    M_raw_columns = df[raw_columns].values
    M_raw_columns[M_raw_columns!=0] = 1./M_raw_columns[M_raw_columns!=0]
    M_raw_columns[M_raw_columns==0] = np.max(M_raw_columns)
    M_raw_columns[M_raw_columns<0] = 0.
    df[raw_columns] = M_raw_columns

def expand_categorical_cols(df, categorical_cols= [('GEOL_REEDBUSH_DOM', 'GEOL_REEDBUSH_DOM_PCT'),
        ('GEOL_HUNT_DOM_CODE', 'GEOL_HUNT_DOM_PCT')]):

    for category, cat_val in categorical_cols:
        dummies = pd.get_dummies(df[category], prefix=category)
        dummies = dummies.multiply(np.expand_dims(df[cat_val].values, axis=1))
        df = df.drop(columns=[category, cat_val])
        df = pd.merge(df, dummies, left_index=True, right_index=True)

    return df

def remove_zero_std_cols(df, cols_for_similarity):

    df_for_similarity = df[cols_for_similarity]
    df_std = np.std(df_for_similarity.values, axis=0)
    zero_std_cols = df_for_similarity.columns.values[(np.isnan(df_std)) | (df_std==0)]
    df.drop(columns=zero_std_cols, inplace=True)
    df_for_similarity.drop(columns=zero_std_cols, inplace=True)
    cols_for_similarity=[x for x in cols_for_similarity if x not in zero_std_cols]

    return cols_for_similarity


def load_gages_dataset(filenames=FILENAMES,
    cols_not_for_similarity=[],
    do_process_raw_columns=True,
    do_expand_categorical_cols=False,
    do_remove_zero_std_cols=True):
    for i,(filename, cols_to_keep) in enumerate(filenames.items()):
        #print(filename)
        if i == 0:
            df = pd.read_csv(os.path.join(DIR, filename), sep=",", encoding = "utf-8", encoding_errors='replace', dtype={"STAID":"str"})
            #print(first_filter(df, cols_to_keep))
            df=df[first_filter(df, cols_to_keep)]
        else:
            df_to_merge = pd.read_csv(os.path.join(DIR, filename), sep=",", encoding = "utf-8", encoding_errors='replace', dtype={"STAID":"str"})
            #print(first_filter(df_to_merge, cols_to_keep))
            df_to_merge=df_to_merge[first_filter(df_to_merge, cols_to_keep)]
            df = df.merge(df_to_merge, left_on='STAID', right_on='STAID')

    if do_process_raw_columns:
        process_raw_columns(df)

    if do_expand_categorical_cols:
        expand_categorical_cols(df)

    cols_for_similarity = [x for x in df.columns.values.tolist() if x not in cols_not_for_similarity]

    if do_remove_zero_std_cols:
        cols_for_similarity = remove_zero_std_cols(df, cols_for_similarity)
    
    return df, cols_for_similarity


if __name__ == '__main__':
    df = load_gages_dataset(FILENAMES)
    
    print(df)

    plot_gauges_in_map(df, column='HUC02')
