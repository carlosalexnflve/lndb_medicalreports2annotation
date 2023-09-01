import numpy as np
import ut2match
import copy

def retif(index):
    # A function to map specific IDs to corresponding arrays
    if index < 6:
        ret = np.array([index])
    elif index == 6:
        ret = np.array([1, 3])
    elif index == 7:
        ret = np.array([2, 5])
    elif index == 8:
        ret = np.array([3, 4, 5])
    elif index == 9:
        ret = np.array([1, 2])
    elif index == 10:
        ret = np.array([1])
    elif index == 11:
        ret = np.array([1, 2])
    return ret

def text2nodules(df, lobe_segmentations, nod_proc_old, count_nod):
    nod_proc = copy.deepcopy(nod_proc_old)
    
    # Correspondence mappings for various attributes
    correspondence1 = np.array([['loc_0', 0], ['unc_0', 1],
                                ['loc_1', 3], ['unc_1', 4],
                                ['loc_2', 6], ['unc_2', 7],
                                ['loc_3', 9], ['unc_3', 10],
                                ['loc_4', 12], ['unc_4', 13],
                                ['loc_5', 15], ['unc_5', 16]])
    
    correspondence2 = np.array([['rem_0', 2], ['rem_1', 5],
                                ['rem_2', 8], ['rem_3', 11],
                                ['rem_4', 14], ['rem_5', 17]])
    
    partLoc = ['LUL', 'LLL', 'RUL', 'ML', 'RLL', 'UL', 'LoL', 'RL', 'LeL',
               'lingula', 'lingula or LLL']

    df_nums = df['num_report']
    list_cts = count_nod[:, 0]
    
    # Filtering out non-common entries
    common_nums = df['num_report'].isin(list_cts)
    not_common_df_nums = np.setdiff1d(df_nums.values, list_cts)
    not_common_list_cts = np.setdiff1d(list_cts, df_nums.values)
    df = df[common_nums]
    
    common_nums = np.intersect1d(df_nums.values, list_cts)
    count_nod = count_nod[np.isin(list_cts, common_nums)]
    nod_proc = [nod_proc[i] for i in np.where(np.in1d(list_cts, common_nums))[0]]
    
    new_nods = len(nod_proc) * ['']
    
    l = len(df)
    fill = np.zeros([l, 18])
    atribution = np.zeros([l, 6])
    atribution2 = np.ones([l, 6]) * -1
    
    # Counters for different scenarios
    count = 0
    fail = 0
    others = 0
    more = 0
    none = 0
    one = 0
    non_anot_true = 0
    non_anot_doubt = 0
    non_anot_doubt2 = 0
    pass_local2 = 0
    pass_size = 0
    pass_charact = 0
    fail2 = 0
    
    for i in range(l):
        # Populating 'fill' array based on attribute presence
        for j in range(12):
            if not ut2match.isNaN(df[correspondence1[j, 0]].values[i]):
                fill[i, int(correspondence1[j, 1])] = 1
        for j in range(6):
            if df[correspondence2[j, 0]].values[i] != ',,,':
                fill[i, int(correspondence2[j, 1])] = 1
        for j in range(6):
            if fill[i, j * 3] == 0 and fill[i, j * 3 + 1] == 0 and fill[i, j * 3 + 2] == 0:
                atribution[i, j] = 1
                atribution2[i, j] = -2
    
    # Processing each entry in the dataframe
    for i in range(l):
        presents = int(6 - sum(atribution[i, :]))
        LNDid = int(df['num_report'].values[i])
        
        if np.size(np.where(count_nod[:, 0] == LNDid)[0]) > 0:
            line = np.where(count_nod[:, 0] == LNDid)[0][0]
            ar_global = []
            
            # Processing each attribute and its information
            for j in range(presents):
                count += 1
                lingula = False
                ar_empty = []
                
                if df[correspondence1[j * 2, 0]].values[i] in partLoc:
                    index = partLoc.index(df[correspondence1[j * 2, 0]].values[i])
                    val_local = retif(index + 1)
                    
                    if df[correspondence1[j * 2, 0]].values[i] == partLoc[9] or df[correspondence1[j * 2, 0]].values[i] == partLoc[10]:
                        lingula = True
                else:
                    val_local = np.array([1, 2, 3, 4, 5])
                
                sums = 0
                text = df[correspondence2[j, 0]].values[i]
                list_text = text.split(",")
                
                for w in range(len(val_local)):
                    sums += count_nod[line, val_local[w]]
                
                if sums == 0:
                    if df[correspondence1[j * 2 + 1, 0]].values[i] == 'is it?' or df[correspondence1[j * 2 + 1, 0]].values[i] == 'how many? is it?':
                        atribution[i, j] = 1
                        atribution2[i, j] = -2
                        non_anot_doubt += 1
                    else:
                        non_anot_true += 1
                        ar_empty.append(j)
                        ar_empty.append('')
                        ar_empty.append(df[correspondence1[j * 2 + 1, 0]].values[i])
                        ar_empty.append(val_local)
                        ar_empty.extend(list_text[1:])
                        new_nods[i] = ut2match.createNewNodules(new_nods[i], count_nod[i, :], ar_empty)
                else:
                    if '' != list_text[0]:
                        k = ut2match.check_local2(LNDid, nod_proc[line], val_local, list_text[0], lingula, lobe_segmentations)
                        if len(k) == 0: pass_local2 += 1
                    else:
                        k = ut2match.not_local2(LNDid, nod_proc[line], val_local, lingula, lobe_segmentations)
                    if len(k) > 0 and ''!= list_text[1]:
                        k = ut2match.check_size(LNDid, nod_proc[line], list_text[1], k)
                        if len(k) == 0: pass_size += 1
                    if len(k) > 0 and ''!= list_text[3]:
                        k = ut2match.check_caract(LNDid, nod_proc[line], list_text[3:], k)
                        if len(k) == 0: pass_charact += 1
                    int_k = len(k)
                    k = ut2match.still_available(nod_proc[line], k)
                    if len(k) == 1:
                        atribution[i,j] = 1
                        atribution2[i,j] = k[0]
                        nod_proc[line][k[0]][20] = j
                        one += 1
                    elif len(k) == 0 and (df[correspondence1[j*2+1,0]].values[i] == 'is it?' or df[correspondence1[j*2+1,0]].values[i] == 'how many? is it?'):
                        atribution[i,j] = 1
                        atribution2[i,j] = -2
                        non_anot_doubt2 += 1

                    elif len(k) > 1:
                        ar_local = []
                        ar_local.append(j)
                        ar_local.append(k)
                        ar_local.append(df[correspondence1[j*2+1,0]].values[i])
                        ar_local.append(val_local)
                        ar_local.extend(list_text[1:])

                        ar_global.append(ar_local)
                        more += 1
                    else:
                        if int_k != len(k): fail2 += 1
                        else: none += 1

                        ar_empty.append(j)
                        ar_empty.append('')
                        ar_empty.append(df[correspondence1[j*2+1,0]].values[i])
                        ar_empty.append(val_local)
                        ar_empty.extend(list_text[1:])

                        new_nods[i] = ut2match.createNewNodules(new_nods[i], count_nod[i,:], ar_empty)
                    
            if len(ar_global) > 0:
                [nod_proc[line], atribution[i, :], atribution2[i, :], fail, new_nods[i]] = ut2match.check_remaining(count_nod[i, :], ar_global, nod_proc[line], atribution[i, :], atribution2[i, :], fail, new_nods[i])

        else:
            others += presents
    
    # Final processing and result packaging
    nod_proc2 = [x + y if isinstance(y, list) else x for x, y in zip(nod_proc, new_nods)]
    dicts = {
        "others": others, "fail": fail, "none": none, "more": more,
        "non_anot_doubt2": non_anot_doubt2, "one": one, "non_anot_true": non_anot_true,
        "non_anot_doubt": non_anot_doubt, "count": count,
        "pass_local2": pass_local2, "pass_size": pass_size,
        "pass_charact": pass_charact, "fail2": fail2
    }
    
    return nod_proc2, not_common_df_nums, not_common_list_cts, atribution, atribution2, dicts
