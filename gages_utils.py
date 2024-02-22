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


NOT_FOR_SIM = ['STAID', 'HUC02','LAT_GAGE','LNG_GAGE','STATE', 'CLASS']


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

def process_raw_columns(df, mapping="C1"):

    raw_columns = [x for x in df.columns if x.startswith("RAW")]
    M_raw_columns = df[raw_columns].values
    if mapping == "C1":
        M_raw_columns[M_raw_columns!=0] = 1./M_raw_columns[M_raw_columns!=0]
        M_raw_columns[M_raw_columns==0] = np.max(M_raw_columns)
        M_raw_columns[M_raw_columns<0] = 0.
    elif mapping == "C0":
        M_raw_columns[M_raw_columns>=0] = np.max(M_raw_columns)-M_raw_columns[M_raw_columns>=0]
        M_raw_columns[M_raw_columns<0] = 0.
        M_raw_columns /= np.max(M_raw_columns)
    
    df[raw_columns] = M_raw_columns


def expand_categorical_cols(df, categorical_cols= [('GEOL_REEDBUSH_DOM', 'GEOL_REEDBUSH_DOM_PCT'),
        ('GEOL_HUNT_DOM_CODE', 'GEOL_HUNT_DOM_PCT')]):

    for category, cat_val in categorical_cols:
        dummies = pd.get_dummies(df[category], prefix=category)
        dummies = dummies.multiply(np.expand_dims(df[cat_val].values, axis=1))
        df = df.drop(columns=[category, cat_val])
        df = pd.merge(df, dummies, left_index=True, right_index=True)

    return df

def load_gages_dataset(filenames=FILENAMES, do_process_raw_columns=False):
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

    df = df.drop(columns=["ASPECT_DEGREES"])
    #df = expand_categorical_cols(df)

    cols_for_similarity = [x for x in df.columns.values.tolist() if x not in NOT_FOR_SIM]
    
    return df, cols_for_similarity



if __name__ == '__main__':
    df = load_gages_dataset(FILENAMES)
    
    #df = df[df["CLASS"]=='Ref']
    #df = df[df["HYDRO_DISTURB_INDX"]==2]

    print(df)

