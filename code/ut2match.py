#import sys
import os
import numpy as np
import random

# Import the function nodEqVol from the readNoduleList module
from readNoduleList import nodEqVol

# Define a function to check if a number is NaN
def isNaN(num):
    return num != num

# Define a function to load data from a file given an idCT and a data folder
def data2file(idCT, datafolder):
     
    fnamesuff = 'fill_new'
    name_file = 'LNDETECTOR-{:04d}/LNDETECTOR-{:04d}_{}.npy'.format(idCT,idCT,fnamesuff)
    datapathlist = os.path.join(os.path.join(datafolder,name_file))    
    file = np.load(datapathlist)    
        
    return file

# Define a function to find the lower limit for the lingula region
def lim_lingula(idCT, datafolder):
    
    file = data2file(idCT, datafolder)
    cd = np.array(np.where(file == 2))
    v = min(cd[0])
    
    return v

# Define a function to filter available nodules based on a certain condition
def still_available(nods, k):
    
    kappas = []
    for s in k:
        if nods[s][20] is False:
            kappas.append(s)
    
    return kappas

# Define a function to filter nodules based on their characteristics and location
def not_local2(idCT, nods, vals, lingula, datafolder):
    
    kappas = []
    for k,nod in enumerate(nods):
        if nod[19] in vals:
            if lingula:
                if nods[k][4] > lim_lingula(idCT, datafolder):
                    kappas.append(k)
            else:
                kappas.append(k)
    
    return kappas

# Define a function to check the local location of nodules
def check_local2(idCT, nods, vals, locPar, lingula, datafolder):
    
    locs2 = ['apical', 'superior', 'basal', 'inferior', 'anterior', 'posterior', 'lateral',
             'medial', 'centrilobular', 'justapleural', 'peripheral']
    
    file = data2file(idCT, datafolder)
    centroid = np.zeros([3,5])
    limits_justa = np.zeros([3,5,2])
    u = np.unique(file)
    
    for cc in range(1,len(u)):
        c = u[cc]
        cd = np.array(np.where(file == c))
        centroid[:,c-1] = 0.5*(np.max(cd,axis = 1) + np.min(cd, axis = 1))
        limits_justa[:,c-1,0] = np.min(cd, axis = 1) + 0.25 * (np.max(cd,axis = 1) - np.min(cd, axis = 1))
        limits_justa[:,c-1,1] = np.min(cd, axis = 1) + 0.75 * (np.max(cd,axis = 1) - np.min(cd, axis = 1))
    
    kappas = not_local2(idCT, nods, vals, lingula, datafolder)

    plans = np.array([[False,False,False],[False,False,False]])
    
    if locs2[4] in locPar:
        plans[0,1] = True
    if locs2[5] in locPar:
        plans[:,1] = True
    if locs2[6] in locPar or locs2[7] in locPar:
        plans[0,2] = True
    if locs2[0] in locPar or locs2[1] in locPar:
        plans[0,0] = True
    if locs2[2] in locPar or locs2[3] in locPar:
        plans[:,0] = True

    kappas2 = []
    for k in kappas:
        if locs2[6] in locPar and nods[k][19] in [1,2]:
            plans[1,2] = True
        elif locs2[7] in locPar and nods[k][19] in [3,4,5]:
            plans[1,2] = True
        
        compare = np.array([
           nods[k][4] > centroid[0,int(nods[k][19]-1)],
           nods[k][5] > centroid[1,int(nods[k][19]-1)],
           nods[k][6] > centroid[2,int(nods[k][19]-1)]])
    
        compare2 = np.array([
            nods[k][4] > limits_justa[0,int(nods[k][19]-1),0] and nods[k][4] < limits_justa[0,int(nods[k][19]-1),1],
            nods[k][5] > limits_justa[1,int(nods[k][19]-1),0] and nods[k][5] < limits_justa[1,int(nods[k][19]-1),1],                    
            nods[k][6] > limits_justa[2,int(nods[k][19]-1),0] and nods[k][6] < limits_justa[2,int(nods[k][19]-1),1]])
        
        check = True
        if plans[0,0]:
            if plans[1,0] != compare[0]:
                check = False
        if plans[0,1]:
            if plans[1,1] != compare[1]:
                check = False
        if plans[0,2]:                            
            if plans[1,2] != compare[2]:
                check = False
        if (~compare2[0] or ~compare2[1] or ~compare2[2]) and (locs2[8] in locPar):
            check = False              
        elif (compare2[1] and compare2[2]) and (locs2[9] in locPar or locs2[10] in locPar):
            check = False
        if check:
            kappas2.append(k)

    return kappas2

def check_size(idCT, nods, locPar, k):
    
    locPar = float(locPar)
    text = ""
    
    kappas = []
    for s in k:
        diam = nods[s][9]
        text = text + str(diam) + ","
        
        if diam <= locPar + 6 and diam >= locPar - 6:
            kappas.append(s)
        elif locPar <= 6 and '' == nods[s][9]:
            kappas.append(s)
    
    return kappas
    
def check_caract(idCT, nods, locPar, k):
    
    n_chars = ['texture','calcification', 'internalStructure', 'lobulation', 'malignancy', 'margin', 'sphericity', 'spiculation', 'subtlety']
    equiv_chars = np.array(range(10,19))
    
    tams = len(locPar)

    text = ""

    kappas = []
    for s in k:
        check = True
        for w in range(tams):
            use = locPar[w].split(':')
            ind = n_chars.index(use[0])

            caract = nods[s][equiv_chars[ind]].split(',')

            try:
                caract = np.array(caract).astype(int)
                if (int(use[1]) not in caract) and (int(use[1])-1 not in caract) and (int(use[1])+1 not in caract):
                    check = False
                    text = text + str(idCT) + "," + str(use) + "," + str(caract)

            except:
                pass
                
        if check:
            kappas.append(s)
            
    return kappas

def check_agree(idCT, nods, k):
    
    agree = np.zeros(len(k))
    
    for it, s in enumerate(k):
        agree[it] = nods[s][8].count("1")
    
    k = np.take(k,np.where(agree == max(agree))[0])
 
    if len(k) > 1:
        agree = np.zeros(len(k))
    
        for it, s in enumerate(k):
            agree[it] = nods[s][8].count(",") + 1
    
        k = np.take(k,np.where(agree == max(agree))[0])
    
    return k
    
def check_agree2(idCT, nods, k, ref):
    
    agree = np.zeros(len(k))
    
    for it, s in enumerate(k):
        agree[it] = nods[s][8].count("1")
    
    k = np.take(k,np.where(agree >= ref)[0])
    
    return k

def check_size2(idCT, nods, locPar, k, positive = False):
    
    kappas = -1
    difs_old = 1000000
    
    for s in k:
        if positive: difs_new = abs(nods[s][9] - float(locPar))

        else: difs_new = float(locPar) - nods[s][9]
        if difs_new < difs_old and positive:
            difs_old = difs_new
            kappas = s
        elif not positive and difs_new < difs_old and difs_new > 0:
            difs_old = difs_new
            kappas = s
            
    if kappas != -1: return [kappas]
    else: return []
    
def check_many(idCT, nods, k, counts, d):
    
    if len(d) == 0:
        k2 = check_agree2(idCT, nods, k, 1)
        if len(k2) == 0:
            k2 = check_agree2(idCT, nods, k, 0)
    elif max(d) == 0:
        k0 = [k[i] for i in range(len(counts)) if counts[i] == 1]
        k2 = check_agree2(idCT, nods, k0, 1)
        if len(k2) == 0:
            k2 = check_agree2(idCT, nods, k0, 0)
    else:
        k1 = [k[i] for i in range(len(counts)) if counts[i] == len(d)+1]
        k0 = [k[i] for i in range(len(counts)) if counts[i] == 1]
        k00 = check_agree2(idCT, nods, k0, 1)
        k11 = check_agree2(idCT, nods, k1, 1)
        if len(k11) > sum(d):
            num_missing = len(k2) - sum(d)
            k2 = random.sample([elem for elem, count in zip(k11, d) if count == 0], num_missing) + k00
        elif len(k00) == 0:
            k2 = check_agree2(idCT, nods, k0, 0)
        else: k2 = k00
        
    return k2

def check_remaining(count_nod, ar, nods, atrib1, atrib2, fail, new_nods):
    
    idCT = count_nod[0]
    for ii, a in enumerate(ar):
    
        k = []
        all_kappas = []
        for i in range(ii,len(ar)):
            ar[i][1] = still_available(nods, ar[i][1])
            all_kappas += ar[i][1]
        all_kappas2 = np.unique(all_kappas)
        all_kappas_count = np.array([all_kappas.count(num) for num in all_kappas2])
        count_dict = dict(zip(all_kappas2, all_kappas_count))
    
        if len(a[1]) > 1 and a[2] != 'how many?' and a[2] != 'how many? is it?':
            # Find the lowest count among the numbers in a1
            lowest_count = min(count_dict[num] for num in a[1])

            # Get the numbers in a1 that have the lowest count
            k = [num for num in a[1] if count_dict[num] == lowest_count]

            if len(k) > 1:
                k = check_agree(idCT, nods, k)
                if len(k) > 1 and ''!= a[4]:
                    k = check_size2(idCT, nods, a[4], k, True)
                if len(k) > 1 and a[5] == 'micro':
                    k = check_size2(idCT, nods, '6', k, False)
                if len(k) > 1 or len(k) == 0:
                    k = check_size2(idCT, nods, '100000', k, False)
        elif len(a[1]) == 1:
            k = a[1]
        elif len(a[1]) > 1:
            
            k = a[1]
            count = [all_kappas_count[i] for i in range(len(all_kappas_count)) if all_kappas2[i] in a[1]]
            
            remaining = len(ar) - ii
            if remaining == 0:
                doubt = []    
            else:
                doubt = np.zeros([remaining])
                for w in range(ii+1,len(ar)):
                    doubt[w-ii-1] = (ar[w][2] == 'is it?')
                    
            k = check_many(idCT, nods, k, count, doubt)      
                    
        if len(k) == 1:
            
            atrib1[a[0]] = 1
            atrib2[a[0]] = k[0]
            nods[k[0]][20] = a[0]

        elif len(k) > 1:

            atrib1[a[0]] = 1
            atrib2[a[0]] = -2
            for w in range(len(k)):
                nods[k[w]][20] = a[0]
                
        elif len(k) == 0 and (a[2] == 'is it?' or a[2] == 'how many? is it?'):

            atrib1[a[0]] = 1
            atrib2[a[0]] = -2
                
        else:
            new_nods = createNewNodules(new_nods, count_nod, a)
            fail += 1
            
    return nods, atrib1, atrib2, fail, new_nods
    
def report2intendedText(df, i, div):

    text = []
    
    fil_df = df[df['num_report'] == i]
    
    correspondence = np.array(['unc_0','rem_0','unc_1','rem_1',
                               'unc_2','rem_2','unc_3','rem_3',
                               'unc_4','rem_4','unc_5','rem_5',])
    
    if isNaN(fil_df[correspondence[div*2]].values[0]):
        text.append("")
    else:
        text.append(fil_df[correspondence[div*2]].values[0])
    
    rem_text = fil_df[correspondence[div*2+1]].values[0].split(",")
    text.extend(rem_text[0:3])
    if len(rem_text[3]) == 1:
        text.append(rem_text[3])
    elif len(rem_text[3:]) > 1:
        text.append(",".join(rem_text[3:]))
    else:
        text.append(rem_text[3])
    
    return text

def calculate_mean_values(a):
    result = []
    for sublist in a:
        if sublist == "":  # Check if sublist is empty
            result.append("")
        elif isinstance(sublist, int):  # Check if sublist is of float type
            result.append(sublist)
        else:
            values = [int(num) for num in sublist.split(',')]
            mean_value = sum(values) / len(values)
            result.append(mean_value)
    return result

# Main function to convert nodules to a final list format
def nod2finalList(nod_proc2, df, nodules_copy):
    name2get = ['LNDbID', 'RadID', 'RadFinding', 'FindingID', 'Nodule', 'x', 'y',
                'z', 'DiamEq_Rad', 'Texture', 'Calcification', 'InternalStructure',
                'Lobulation', 'Malignancy', 'Margin', 'Sphericity', 'Spiculation',
                'Subtlety', 'Lobe',  'TextInstanceID', 'TextQuestion', 'Pos_Text',
                'Diam_Text', 'NodType', 'Caract_Text', 'Where']
    
    nod_proc3 = []
    nod_proc3.append(name2get)
    for l in range(len(nod_proc2)):
        for c in range(len(nod_proc2[l])):
            if isinstance(nod_proc2[l][c][3], (int)):
                for w in range(len(nodules_copy)):
                    if (nodules_copy[w][0] == nod_proc2[l][c][0]) and (nodules_copy[w][3] == nod_proc2[l][c][3]):
                        break
            list_item = []

            list_item.extend(nod_proc2[l][c][0:4])
            list_item.append(nod_proc2[l][c][8])
                       
            if nod_proc2[l][c][20] is False:
                list_item.extend(nodules_copy[w][4:7])
                list_item.append(nod_proc2[l][c][9])
                list_item.extend(calculate_mean_values(nod_proc2[l][c][10:19]))
                list_item.append(nod_proc2[l][c][19])
                list_item.append("")
                list_item.append("")
                list_item.append("")
                list_item.append("")
                list_item.append("")
                list_item.append("")
                list_item.append("RadAnnotation")
            elif nod_proc2[l][c][2] is "":
                list_item.append("")
                list_item.append("")
                list_item.append("")
                list_item.append(nod_proc2[l][c][9])
                list_item.extend(calculate_mean_values(nod_proc2[l][c][10:19]))
                list_item.extend(nod_proc2[l][c][19:21])
                list_item.extend(report2intendedText(df, nod_proc2[l][c][0], nod_proc2[l][c][20]))
                if len(nod_proc2[l][c][21]) > 0:
                    list_item.append("TextReport (" + nod_proc2[l][c][21] + ")")
                else:
                    list_item.append("TextReport")
            elif isinstance(nod_proc2[l][c][20], (int)):
                list_item.extend(nodules_copy[w][4:7])
                list_item.append(nod_proc2[l][c][9])
                list_item.extend(calculate_mean_values(nod_proc2[l][c][10:19]))
                list_item.extend(nod_proc2[l][c][19:21])
                list_item.extend(report2intendedText(df, nod_proc2[l][c][0], nod_proc2[l][c][20]))
                if list_item[9] == "":
                    list_item[9] = 5
                    list_item.append("TextReport+RadAnnotation (created: texture)")
                else:
                    list_item.append("TextReport+RadAnnotation")
    
            nod_proc3.append(list_item)
            
    return nod_proc3

def convert_created(text):
    if not text:  # Check if the list is empty
        return ''  # Return an empty string

    return 'created: ' + ', '.join(text)

def createNewNodules(new_nods, count_nod, a):

    name2get = ['texture', 'calcification', 'internalStructure',
                'lobulation', 'malignancy', 'margin', 'sphericity', 'spiculation',
                'subtlety']
    
    created = []
    
    num_nods = 1
    if a[2] == "how many?":
        num_nods = 2
        
    for i in range(num_nods):
        nod = ['']*22
        nod[0] = int(count_nod[0])
        nod[3] = int(np.sum(count_nod[1:]) + len(new_nods) + 1)
        
        nod[19] = a[3] if len(a[3]) > 1 else a[3][0]
        nod[20] = a[0]
        
        if ''!= a[4]:
            nod[9] = float(a[4])
        elif a[5] == 'micro':
            nod[9] = 2
            if i != 1: created.append('size')
        else:
            nod[9] = 4
            if i != 1: created.append('size')
        
        texture = False
        for aa in a[6:]:
            if aa != '':
                caract = aa.split(":")
                ind = name2get.index(caract[0])
                nod[ind+10] = int(caract[1])
                if nod[ind+10] == '' and caract[0] == 'calcification':
                    nod[10] = 5
                    texture = True
                elif caract[0] == 'texture':
                    texture = True
        if num_nods > 1 and nod[9] < 5.75 and nod[10] == '':
            nod[10] = 5
        elif nod[9] > 6 and nod[10] == '':
            nod[10] = 5
        else:
            nod[10] = 1

        if texture is False:
            if i != 1: created.append('texture')
        
        nod[21] = convert_created(created)
        try:
            new_nods.append(nod)
        except:
            new_nods = [nod]
        
    return new_nods

def get_caract(text, other, caract ='texture'):
    ca2points = caract + ":"
    if ca2points in text:
        # Split the text by commas
        parts = text.split(",")
        
        for part in parts:
            if ca2points in part:
                # Extract the texture value
                texture = part.split(":")[1].strip()
                return int(texture)
    
    # If no texture value is found, return the value of 'other'
    return other

def substitute2Fleishcner(nod_p,form='RadAnnotation'):
    
    nod_new = []
    first = ['LNDbID','FindingID','Nodule','Volume','Text','Where']
    nod_new.append(first)
    
    for i in range(1,len(nod_p)):
        if (form == 'RadAnnotation') and ('RadAnnotation' in nod_p[i][25]):
            nod_items = []
            nod_items.append(nod_p[i][0])
            nod_items.append(nod_p[i][3])
            nod_items.append(1 if '1' in nod_p[i][4] else 0)
            nod_items.append(nodEqVol(nod_p[i][8]))
            nod_items.append(nod_p[i][9])
            nod_items.append(nod_p[i][25])
            nod_new.append(nod_items)
        elif (form == 'TextReport') and ('TextReport' in nod_p[i][25]):
            nod_items = []
            nod_items.append(nod_p[i][0])
            nod_items.append(nod_p[i][3])
            nod_items.append(1)
            nod_items.append(nodEqVol(float(nod_p[i][22]) if nod_p[i][22] != "" else float(nod_p[i][8])))
            nod_items.append(get_caract(nod_p[i][24], nod_p[i][9]))
            nod_items.append(nod_p[i][25])
            nod_new.append(nod_items)
            
    return nod_new
    