#mit_bih_util.py>

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import wfdb
import datetime
import json
from collections import Counter

# MIT-BIH anotacijų pažymėjimai:

# Papildomos anotacijos
invalid_beat = [
    "[", "!", "]", "x", "(", ")", "p", "t", 
    "u", "`", "'", "^", "|", "~", "+", "s", 
    "T", "*", "D", "=", '"', "@"
]

# Aritminės anotacijos
abnormal_beats = [
    "L", "R", "B", "A", "a", "J", "S", "V", 
    "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"
]

# def get_key_list(dictionary):
#     key_list = [key for (key, value) in dictionary.items()]
#     return key_list
# Panaikintas -  keisti į dictionary.keys()
# https://www.w3schools.com/python/ref_dictionary_keys.asp


def get_rev_dictionary(dictionary):
    rev_dict = {value : key for (key, value) in dictionary.items()}
    return rev_dict


def get_symbol_list(atr_symbols, atr_samples, seq_start, seq_end):
    # Surenkame išpjautos EKG sekos anotacijas ir jų indeksus sekoje
    # ir patalpiname sąraše.
    beat_locs = []
    beat_symbols = []

    for i in range(len(atr_samples)):
        if atr_samples[i] > seq_start and atr_samples[i] < seq_end:
            beat_symbols.append(atr_symbols[i])
            beat_locs.append(atr_samples[i]-seq_start)   
            # beat_locs.append(atr_samples[i])   

    return (beat_symbols,beat_locs)


def beat_convert_to_string(beat_symbols,beat_locs):
# Paverčiame anotacijų sąrašą į tekstą (string) įrašymui į csv failą.
    symb_string = ','.join(beat_symbols)
    # Paverčiame anotacijų indeksų sąrašą į tekstą (string) 
    # įrašymui į csv failą
    lst = [str(i) for i in beat_locs]
    locs_string = ','.join(lst)
    return(symb_string, locs_string)


def get_seq_start_end(signal_length,i_sample,window_left_side,window_right_side):
    # Nustatome išskiriamos EKG sekos pradžią ir pabaigą
    seq_start = i_sample - window_left_side
    seq_end = i_sample + window_right_side
    if (seq_start < 0 or seq_end > signal_length):
        print("\nseq_start: ", seq_start, " seq_end: ", seq_end)
        return (None,None)
    else:    
        return (seq_start, seq_end)

def load_ecg_seq_raw(idx, descr, db_path):
# Formuojame ECG seką iš paciento įrašo, saugomo .mat formatu 
# Surandame sekos descriptorių pagal užduotą idx

    descr_seq = descr.loc[idx]
    subject = descr_seq['subject']

    # Nuskaitome visą paciento EKG įrašą
    rec = wfdb.rdrecord(f'{db_path}/{subject}')
    # print(f'{db_path}/{subject}')

    # Paliekama tik 1-a derivacija
    signal = rec.p_signal[:,0] # 1D -> 2D, sklearn expects 2D array
    # sequence = np.reshape(signal,(-1,1))
    
    # Išskiriame seką ir gražiname
    seq_start  = descr_seq['seq_start']
    seq_end  = descr_seq['seq_end']
    sequence = signal[seq_start:seq_end]
    # gražiname seką 1d: sequence.shape(seq_end-seq_start,)
    return sequence

def load_ecg_seq_raw2(descr_seq, db_path):
# Formuojame ECG seką iš paciento įrašo, saugomo .mat formatu 
# Naudojame užduotą sekos descriptorių

    subject = descr_seq['subject']

    # Nuskaitome visą paciento EKG įrašą
    rec = wfdb.rdrecord(f'{db_path}/{subject}')
    # print(f'{db_path}/{subject}')

    # Paliekama tik 1-a derivacija
    signal = rec.p_signal[:,0]
    
    # 1D -> 2D, sklearn expects 2D array
    # sequence = np.reshape(signal,(-1,1))
    
    # Išskiriame seką ir gražiname
    seq_start  = descr_seq['seq_start']
    seq_end  = descr_seq['seq_end']
    sequence = signal[seq_start:seq_end]
    # gražiname seką 1d: sequence.shape(seq_end-seq_start,)
    return sequence

def load_ecg_seq2(descr_seq, db_path):
# Formuojame ECG seką iš paciento įrašo, saugomo .npy formatu 
# Naudojame užduotą sekos descriptorių

    subject = descr_seq['subject']

    # Nuskaitome visą paciento EKG įrašą
    file_path=os.path.join(db_path, str(subject) + ".npy")
    signal = np.load(file_path)

    # Išskiriame seką ir gražiname
    seq_start  = descr_seq['seq_start']
    seq_end  = descr_seq['seq_end']
    sequence = signal[seq_start:seq_end]
    # gražiname seką 2d: sequence.shape(seq_end-seq_start, 1)
    return sequence

def load_ecg_seq(idx, descr, db_path):
# Formuojame ECG seką iš paciento įrašo, saugomo .npy formatu 
# Surandame sekos descriptorių pagal užduotą idx

    descr_seq = descr.loc[idx]
    subject = descr_seq['subject']

    # Nuskaitome visą paciento EKG įrašą
    file_path=os.path.join(db_path, str(subject) + ".npy")
    signal = np.load(file_path)

    # Išskiriame seką ir gražiname
    seq_start  = descr_seq['seq_start']
    seq_end  = descr_seq['seq_end']
    sequence = signal[seq_start:seq_end]
    # gražiname seką 2d: sequence.shape(seq_end-seq_start, 1)
    return sequence

def show_ecg_seq(id, descr, sequence):
# Atvaizduoja grafiškai EKG seką 
    descr_seq = descr.loc[id]

    subject = descr_seq['subject']
    seq_nr = descr_seq['seq_nr']

    # Seką atvaizduojame
    seq_start  = descr_seq['seq_start']
    seq_end  = descr_seq['seq_end']
    fs = descr_seq['fs']
    start = convert(round(seq_start/fs))
    end = convert(round(seq_end/fs))
    annot_symbol = descr_seq['symbol']
    class_label = descr_seq['label']
    loc = descr_seq['i_sample'] - seq_start

    # print("annot.=", annot_symbol," class=", class_label, " loc=", loc)
    title = "MIT-BIH Record {} seq. nr {} ".format(subject,seq_nr)
    wfdb.plot_items(signal=sequence, title=title, time_units='samples', figsize=(14,5))


def show_ecg_seq_annot(id, descr, sequence):
# Išveda į ekraną EKG sekos anotacijas ir jų indeksus. Naudojamas laikinai, kol nebus anotacijos atvaizduotos grafiškai
    descr_seq = descr.loc[id]
    # Sekos pradžia, pabaiga
    seq_start  = descr_seq['seq_start']
    seq_end  = descr_seq['seq_end']
    fs = descr_seq['fs']
    start = convert(round(seq_start/fs))
    end = convert(round(seq_end/fs))
    print("start= {} ({}) end= {} ({})".format(start,seq_start,end,seq_end))

    annot_symbol = descr_seq['symbol']
    class_label = descr_seq['label']
    loc = descr_seq['i_sample'] - seq_start
    symb_string = descr_seq['beat_symbols']
    locs_string = descr_seq['beat_locs']
    beat_symbols, beat_locs = beat_convert_to_list(symb_string, locs_string)
    symb_string = '     '.join(beat_symbols)
    supl = '    '
    symb_string = supl + symb_string 
    print(symb_string)
    le = len(beat_symbols)
    for i in range(le): print("%6d" % beat_locs[i], end='')


def convert(n): 
    return str(datetime.timedelta(seconds = n))


def beat_convert_to_list(symb_string, locs_string):
    # Iškoduojamas iš teksto pavidalu (string) saugomo EKG sekos anotacijų sąrašas
    beat_symbols = symb_string.split(',')
    lst = locs_string.split(',')
    beat_locs = [int(i) for i in lst] 
    # Atverčiamas iš užkoduoto teksto pavidalu (string) EKG sekos
    # anotacijų indeksų sąrašas
    return(beat_symbols,beat_locs)


def get_seq_id(subject, seq_nr, descr):
# Nurodytam subject ir seq_nr, EKG segmentų masyve
# surandamas įrašas ir gražinamas jo indeksas id
# Jei įrašo neranda, gražina None.
# Jei suranda daugiau, gražina pirmą.
#   
    obj = descr.index[(descr['subject']==subject) & (descr['seq_nr']==seq_nr)]
    lst = obj.tolist()
    if (len(lst) == 0):
        return None
    else:    
        return lst[0]

def get_file_name(descr_seq):
    subject = descr_seq["subject"]
    seq_nr = descr_seq["seq_nr"]
    label_symb = descr_seq["label_symb"]
    symbol = descr_seq["symbol"]
    file_name = str(subject) + "_" + str(seq_nr) + "_" + symbol+label_symb
    return file_name

# Pretendentas išmęsti
# //////////////////////////////////////////////////////////////////
def read_duom_info(SCHEMA_PATH):
# Nuskaitoma ir išvedama informacija apie duomenis ir hyperparametrus
    file_name = os.path.join(SCHEMA_PATH,'schema_info.json')
    with open(file_name) as json_file:
        info = json.load(json_file)
    return info

def show_duom_info(info):
    schema = info['schema']
    selected_beats = info['selected_beats']
    name_db = info['name_db']
    db_path = info['db_path']
    fs = info['fs']
    window_left_side = info['window_left_side']
    window_right_side = info['window_right_side']
    posl = info['posl']
    window_length = window_left_side+window_right_side
    subj_list = info['subj_list']
    all_data_dir = info['all_data_dir'] 
    raw_dir = info['raw_dir'] 
    image_dir = info['image_dir'] 

    print("\nDuomenys: ", name_db, "\nDuomenų aplankas: ",db_path)
    print("\nPacientų įrašų sąrašas:\n",subj_list)
    print("\nDiskretizavimo dažnis: ", fs)
    print("\nKLASIFIKAVIMO SCHEMA: ", schema)
    print("Schemos klasės:\n",selected_beats)
    if (all_data_dir != None and raw_dir != None):
        data_path = os.path.join(all_data_dir, raw_dir)
    else:
        data_path = None    
    print("\nDuomenų aplankas: ",data_path)
    if (all_data_dir != None and image_dir != None):
        image_path = os.path.join(all_data_dir, image_dir)
    else:
        image_path = None    
    print("Vaizdų aplankas: ",image_path)

    # Išvedamas EKG sekų lango plotis
    print("\nEKG sekų lango plotis:")
    print("kairė pusė: ", window_left_side, "dešinė pusė: ", window_right_side)
    window_length = window_left_side + window_right_side 
    print("Visas EKG sekos ilgis: ", window_length,"\n")
    return 

def visual_descr(descr, flag):
# Išveda į ekraną EKG sekų deskriptorių reikšmes
# Jei flag == True, tai išveda kartu su klasifikatoriaus priskirtomis klasės reikšmėmis pred ir atitinkamu klasės simboliu pred_symbol 
    if (flag == True):
        print("| id"+" "*4+"| subj"+" "*1+"| seq"+" "*3+"| lbl"+" "*1+"| smb"+" "*1+"| pred"+" "*1+"| smb"+" "*1+"| beats")
        row = "| {id:<5d} | {subject:<4d} | {seq_nr:<6d}| {label:<4d}| {symbol:3s} | {pred:<5d}| {pred_symb:<4s}| {beat_str:<6s}".format
    else:
        print("| id"+" "*4+"| subj"+" "*1+"| seq"+" "*3+"| lbl"+" "*1+"| smb"+" "*1+"| beats")
        row = "| {id:<5d} | {subject:<4d} | {seq_nr:<6d}| {label:<4d}| {symbol:3s} | {beat_str:<6s}".format
    
    for id in descr.index:
        subject = descr.loc[id, "subject"]
        seq_nr = descr.loc[id, "seq_nr"]
        beat_symbols = descr.loc[id, "beat_symbols"]
        label = descr.loc[id, "label"]
        symbol = descr.loc[id, "symbol"]
        pred = descr.loc[id,"pred"]
        pred_symb = descr.loc[id,"pred_symb"]
        if (flag == True):
            print(row(id=id, subject=subject, seq_nr=seq_nr, label=label, symbol=symbol, pred=pred, pred_symb=pred_symb, beat_str=beat_symbols))
        else:
            print(row(id=id, subject=subject, seq_nr=seq_nr, label=label, symbol=symbol, beat_str=beat_symbols))

def visual_descr_row(id, descr_row, flag):
# Išveda į ekraną EKG sekų deskriptorių reikšmes
# Jei flag == True, tai išveda kartu su klasifikatoriaus priskirtomis klasės reikšmėmis pred ir atitinkamu klasės simboliu pred_symbol 
    if (flag == True):
        print("| id"+" "*4+"| subj"+" "*1+"| seq"+" "*3+"| lbl"+" "*1+"| smb"+" "*1+"| pred"+" "*1+"| smb"+" "*1+"| beats")
        row = "| {id:<5d} | {subject:<4d} | {seq_nr:<6d}| {label:<4d}| {symbol:3s} | {pred:<5d}| {pred_symb:<4s}| {beat_str:<6s}".format
    else:
        print("| id"+" "*4+"| subj"+" "*1+"| seq"+" "*3+"| lbl"+" "*1+"| smb"+" "*1+"| beats")
        row = "| {id:<5d} | {subject:<4d} | {seq_nr:<6d}| {label:<4d}| {symbol:3s} | {beat_str:<6s}".format
    
    subject = descr_row["subject"]
    seq_nr = descr_row["seq_nr"]
    beat_symbols = descr_row["beat_symbols"]
    label = descr_row["label"]
    symbol = descr_row["symbol"]
    pred = descr_row["pred"]
    pred_symb = descr_row["pred_symb"]
    
    if (flag == True):
        print(row(id=id, subject=subject, seq_nr=seq_nr, label=label, symbol=symbol, pred=pred, pred_symb=pred_symb, beat_str=beat_symbols))
    else:
        print(row(id=id, subject=subject, seq_nr=seq_nr, label=label, symbol=symbol, beat_str=beat_symbols))


def load_dict(path):
    # reading the data from the file
    with open(path) as f:
        data = f.read()
        # reconstructing the data as a dictionary
        js = json.loads(data)
        f.close()
    return(js)

def json_write(file, dir, file_name):
    # Įrašome json failą į diską 
    file_path = os.path.join(dir, file_name)
    with open(file_path, 'w') as outfile:
        json.dump(file, outfile)

def create_dir(parent_dir):
    # Sukuriami rekursyviškai aplankai, jei egzistuoja - tai nekuria
    # https://smallbusiness.chron.com/make-folders-subfolders-python-38545.html

    try:
        os.makedirs(parent_dir)
        print("Directory '%s' created successfully" % parent_dir)
        # print("Directory {:s} created successfully".format(parent_dir)
    except OSError as error:
        print("Directory '%s' already exists" % parent_dir)


def create_subdir(parent_dir, names_lst):
    # Sukuriami subdirektoriai su nurodytais pavadinimais
    for name in names_lst:
        # sukuriami aplankai EKG sekų vaizdams
        sub_dir = os.path.join(parent_dir, name)
        try:
            os.makedirs(sub_dir)
            print("Directory '%s' created successfully" % sub_dir)
        except OSError as error:
            print("Directory '%s' already exists" % sub_dir)


def seq_image_to_disk(seq1d, seq_nr, subject_dir, symbol):
    # Suformuojame vaizdą
    x = np.arange(0, len(seq1d), 1)
    fig = plt.figure(figsize=(6,3))
    plt.plot(x, seq1d, color="#6c3376", linewidth=2)
    
    # Įrašome į diską
    image_folder = os.path.join(subject_dir, symbol)
    if (os.path.exists(image_folder) == False):
        print('Klaida! ', image_folder,' neegzistuoja')
    file_name = str(seq_nr) + "_" + symbol + ".png" 
    file_path = os.path.join(image_folder, file_name)
    
    # Įrašome į atitinkamą sub-aplanką
    plt.savefig(file_path)
    plt.close()
    # print(file_path)

def set_seq_start_end(signal_length,i_sample,window_left_side,window_right_side):
    # Nustatome išskiriamos EKG sekos pradžią ir pabaigą.
    #  Variantas, kai langas gali būti koreguojamas
    seq_start = i_sample - window_left_side
    if (seq_start < 0):
        seq_start = 0

    seq_end = i_sample + window_right_side
    if (seq_end > signal_length):
        seq_end = signal_length

    return (seq_start, seq_end)
        
def get_subj_seq_nr(seq_file_name):
    # pvz. seq_file_name = '210_255_NN'
    lst = seq_file_name.split('_',2)
    subject = int(lst[0])
    seq_nr = int(lst[1])
    return subject, seq_nr

def show_seq_ext(db_path, subject, i_sample, win_ls, win_rs, win_ls_ext, win_rs_ext):
# Išpjauna užduoto ilgio seką iš MIT-BIH įrašo ir sukuria jos vaizdą su anotacijomis

# db_path - paciento EKG įrašų aplankas
# subject - paciento EKG įrašo numeris
# i_sample - R dantelio, kurio atžvilgiu formuojama seka, indeksas viso EKG įrašo reikšmių masyve
# win_ls - klasifikuojamo EKG segmento plotis iki R pūpsnio (iš kairės) 
# win_rs - klasifikuojamo EKG segmento plotis nuo R pūpsnio (iš dešinės)
# win_ls_ext - vaizduojamo EKG segmento plotis iki R pūpsnio (iš kairės) 
# win_rs_ext - vaizduojamo EKG segmento plotis už R pūpsnio (iš dešinės) 

    ax = plt.gca()
    
    # Nuskaitome paciento įrašo anotacijų failą
    annotation = wfdb.rdann(f'mit-bih-arrhythmia-database-1.0.0/{subject}', 'atr')

    # Įrašo anotacijos ir jų lokacijos EKG įraše
    atr_symbol = annotation.symbol
    atr_sample = annotation.sample

    # Nuskaitome visą paciento EKG įrašą
    rec = wfdb.rdrecord(f'{db_path}/{subject}')

    # Paliekama tik 1-a derivacija
    signal = rec.p_signal[:,0]
    # 1D -> 2D, sklearn expects 2D array
    # sequence = np.reshape(signal,(-1,1))
        
    signal_length = rec.p_signal.shape[0]

    # surandame užduoto ilgio sekos pradžią ir pabaigą,
    # jei reikia - koreguojame
    seq_start, seq_end = set_seq_start_end(signal_length,i_sample,win_ls_ext,win_rs_ext)

    # Išskiriame seką
    sequence = signal[seq_start:seq_end]
    # gražiname seką 2d: sequence.shape(seq_end-seq_start, 1)

    # suformuojame anotacijų žymes
    beat_symbols,beat_locs = get_symbol_list(atr_symbol,atr_sample, seq_start, seq_end)

    # deltax ir deltay simbolių pozicijų koregavimui
    min = np.amin(sequence)
    max = np.amax(sequence)
    deltay = (max - min)/20
    deltax = len(sequence)/100

    # suformuojame vaizdą
    x = np.arange(0, len(sequence), 1)
    ax.plot(x, sequence, color="#6c3376", linewidth=2)
    left_mark = i_sample - seq_start - win_ls
    right_mark = i_sample - seq_start + win_rs
    ax.axvline(x = left_mark, color = 'b', linestyle = 'dotted')
    ax.axvline(x = right_mark, color = 'b', linestyle = 'dotted')
    for i in range(len(beat_locs)):
        ax.annotate(beat_symbols[i],(beat_locs[i]-deltax,sequence[beat_locs[i]]+deltay))
    ax.set_ylim([min, max+2*deltay])
    return(ax)

