import os
from pathlib import Path
import pandas as pd
import numpy as np
import neurokit2 as nk
import json

def read_json_file(fpath):
    try:
        with open(fpath) as f:
            data_js = json.load(f)
            return data_js
    except FileNotFoundError:
        return {"error": "File not found"}

def runtime(s):
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    print('Runtime: {:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds)))

def zive_read_file_1ch(filename):
    f = open(filename, "r")
    a = np.fromfile(f, dtype=np.dtype('>i4'))
    ADCmax=0x800000
    Vref=2.5
    b = (a - ADCmax/2)*2*Vref/ADCmax/3.5*1000
    ecg_signal = b - np.mean(b)
    return ecg_signal

def collect_filenames(db_path):
    # - surenkame aplankų ir failų sąrašą
    # https://careerkarma.com/blog/python-list-files-in-directory/

    # Walking a directory tree and printing the names of the directories and files
    filepaths =[]

    # Surandame buferyje visus failus, kurie neturi extension json ar csv
    for dirpath, dirnames, files in os.walk(db_path):
        # https://linuxhint.com/python-os-walk-example/
        # print(dirpath, dirnames, files)
        for file_name in files:
            root, extension = os.path.splitext(file_name)
            if (extension == '.json' or extension == '.csv'):
                continue
            else:
                filepaths.append(str(file_name))
    return filepaths


def collect_list(db_path):
# Analogiškas, kaip MAKETE, tačiau be papildomos informacijos iš comment.csv prijungimo

    # - paruošiame dataframe įrašų sužymėjimui
    df_list = pd.DataFrame({
                            'file_name': pd.Series(dtype='str'),
                            'userId': pd.Series(dtype='str'),
                            'recordingId': pd.Series(dtype='str'),
                            'N': pd.Series(dtype='int'),
                            'S': pd.Series(dtype='int'),
                            'V': pd.Series(dtype='int'),
                            'U': pd.Series(dtype='int'),
                            'Tr': pd.Series(dtype='int'),
                            'incl': pd.Series(dtype='int'), 
                            'flag': pd.Series(dtype='int'),
                            'comment': pd.Series(dtype='str'),
                            'recorded_at':pd.Series(dtype='datetime64[ns]')})
    df_list = pd.DataFrame()

    # - surenkame aplankų ir failų sąrašą
    # https://careerkarma.com/blog/python-list-files-in-directory/

    # Walking a directory tree and printing the names of the directories and files
    filepaths =[]
    new_files = 0

    # # Surandame buferyje visus failus, kurie neturi extension json
    # for dirpath, dirnames, files in os.walk(db_path):
    filenames = collect_filenames(db_path)

    # Einame per visus įrašus, įtraukiame įrašo parametrus į sąrašą
    for filename in filenames:
        timestamp = int(filename.replace('.', ''))
        filename_js = filename + '.json'    
        filepath = Path(db_path, filename_js) 
        with open(filepath,'r', encoding='UTF-8', errors = 'ignore') as f:
            data = json.loads(f.read())
        dict_annot = data['rpeakAnnotationCounts']
        # filename_str = "{:.3f}".format(filename)
        # print(filename, filename_str)

        dict_row = {'file_name':filename, 'userId':data['userId'], 'recordingId':data['recordingId'],
        'N':dict_annot.get('N',0), 'S':dict_annot.get('S',0), 'V':dict_annot.get('V',0),
        'U':dict_annot.get('U',0), 'Tr':len(data['noises']),'incl':0,'flag':0,'comment':"",'recorded_at':timestamp}
        
        df_row = pd.DataFrame([dict_row])
        df_list = pd.concat([df_list,df_row], axis=0) 
        new_files +=1

    if (new_files != 0):
        df_list['recorded_at'] =  pd.to_datetime(df_list['recorded_at'], unit='s')
        df_list = df_list.convert_dtypes() # koreguoju dtypes į geriau atitinkančius turinį
        # print("df_list:", df_list)
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.convert_dtypes.html

    # Jei db_path yra failas su papildoma informacija, jį nuskaitome ir papildome df_list su šia informacija
    # add_comments(df_list,comments), papildoma informacija yra csv pavidalu
    
    dtypes = {
        'N': int,
        'S': int,
        'V': int,
        'U': int,
        'Tr':int,
        'incl': int,
        'flag': int,
        'comment': str }
    
    df_list = df_list.astype(dtypes)

    # file_path = Path(db_path, 'comments_new.csv')
    # if (file_path.exists()):
    #     print("ieškome comments.csv")
    #     df_comments = pd.read_csv(file_path, dtype=dtypes)
        # print(df_comments)
        # print(df_comments.dtypes)

# Čia reikia įdėti file_name sutvarkymą - reikia paversti, kad būtų visi trys ženklai plėtinyje
        # for i in range(len(df_comments)):
        #     df_comments.loc[i, 'file_name'] =  "{:.3f}".format(df_comments.loc[i, 'file_name'])
        # df_list_commented = add_comments(df_list, df_comments)
        # df_list_commented = df_list_commented.astype(dtypes)
        # df_list_commented['incl'] = df_list_commented['incl'].astype(int)
        # df_list_commented['flag'] = df_list_commented['flag'].astype(int)
        # print("df_list_commented:\n", df_list_commented)
       
    # Įrašome df_list į failą 'list.json'
    # filepath = Path(db_path,'list.json')
    # df_list_commented.to_json(filepath, orient = 'table', index=False)
    # print(f'Failų sąrašas įrašytas į:  {filepath}')

    return(df_list)
