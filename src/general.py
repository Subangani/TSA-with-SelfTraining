import csv


def dict_update(original, temp):
    """
    This will update original dictionary key, and values by comparing with temp values
    :param original:
    :param temp:
    :return: original updated dictionary and a success statement
    """
    is_success = False
    result = {}
    for key in temp.keys():
        global_key_value = original.get(key)
        local_key_value = temp.get(key)
        if key not in original.keys():
            result.update({key:local_key_value})
        else:
            result.update({key: local_key_value + global_key_value})
    original_temp = original.copy()
    result.update(original_temp)
    return result,is_success


def get_scores():
    try:
        testedFile="../dataset/test.csv.svm_result"
        f0=open(testedFile,"r")
        line = f0.readline()
        readers = csv.reader(f0)
        TP = TN = TNeu = FP_N = FP_Neu = FN_P = FN_Neu = FNeu_P = FNeu_N = 0
        dic = {'positive': 2.0, 'negative': -2.0, 'neutral': 0.0}
        for line in readers:
            new = str(line[2])
            old = str(dic.get(line[0]))
            if old == new:
                if new == "2.0":
                    TP += 1
                elif new == "-2.0":
                    TN += 1
                elif new == "0.0":
                    TNeu += 1
            else:
                if new == "2.0" and old == "-2.0":
                    FP_N += 1
                elif new == "2.0" and old == "0.0":
                    FP_Neu += 1
                elif new == "-2.0" and old == "2.0":
                    FN_P += 1
                elif new == "-2.0" and old == "0.0":
                    FN_Neu += 1
                elif new == "0.0" and old == "2.0":
                    FNeu_P += 1
                elif new == "0.0" and old == "-2.0":
                    FNeu_N += 1

        accuracy = get_divided_value((TP + TN + TNeu),
                                     (TP + TN + TNeu + FP_N + FP_Neu + FN_P + FN_Neu + FNeu_P + FNeu_N))
        pre_p = get_divided_value(TP, (FP_N + FP_Neu + TP))
        pre_n = get_divided_value(TN, (FN_P + FN_Neu + TN))
        pre_neu = get_divided_value(TNeu, (FNeu_P + FNeu_N + TNeu))
        re_p = get_divided_value(TP, (FN_P + FNeu_P + TP))
        re_n = get_divided_value(TN, (FP_N + FNeu_N + TN))
        re_neu = get_divided_value(TNeu, (FNeu_P + FNeu_N + TNeu))
        f_score_p = 2 * get_divided_value((re_p * pre_p), (re_p + pre_p))
        f_score_n = 2 * get_divided_value((re_n * pre_n), (re_n + pre_n))
        f_score_average = round((f_score_p + f_score_n) / 2, 4)
        print accuracy, pre_p, pre_n, pre_neu, re_p, re_n, re_neu, f_score_p, f_score_n, f_score_average
    except TypeError:
        print TypeError.message


def get_divided_value(numerator, denominator):
    if denominator == 0:
        return 0.0
    else:
        result = numerator/(denominator * 1.0)
    return round(result, 4)


def temp_difference_cal(time_list):
    """
    This function is used when a set of time values are added and difference between last two are obtained
    :param time_list:
    :return: difference
    """
    if len(time_list) > 1:
        final = float(time_list[len(time_list) - 1])
        initial = float(time_list[len(time_list) - 2])
        difference = final - initial
    else:
        difference = -1.0
    return difference
