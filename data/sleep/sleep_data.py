import os

import numpy as np
import pandas

from utils.pre_processing import formatted_data, missing_proportion, get_train_median_mode


# TODO One hot encode categorical variables
# TODO some variables are negative date
# https://sleepdata.org/datasets/shhs
# https://sleepdata.org/datasets/shhs/variables

# dropped <- c("nsrrid", "pptid", "any_cvd", "status", "folder")

# myglmnet = cv.glmnet(as.matrix(myData[-test_idx, ]),
# Surv(new_dat[-test_idx, "status"], new_dat[-test_idx, "any_cvd"]), family="cox", alpha=1, nfolds=10)

# myData <- new_dat[, which(colnames(new_dat) %in% new_temp$id[c(grep('Demographics',
# new_temp$folder), grep('Medications', new_temp$folder),
# grep('Questionnaires', new_temp$folder))])]

# Surv (time, event), time = status, event = any_cvd,

## categorical variables
# $2
# [1] "gender"    "imsbpae"   "imdbpae"   "ursbpae"   "urdbpae"   "skrctnae"  "tripae"    "othprbae"
#  [9] "evsmok15"  "smknow15"  "asa15"     "estrgn1"   "progst1"   "htnmed1"   "anar1a1"   "lipid1"
# [17] "ohga1"     "insuln1"   "sympth1"   "tca1"      "asa1"      "nsaid1"    "benzod1"   "premar1"
# [25] "pdei1"     "ntca1"     "warf1"     "loop1"     "hctz1"     "hctzk1"    "ccbir1"    "ccbsr1"
# [33] "alpha1"    "anar1b1"   "anar1c1"   "anar31"    "pvdl1"     "basq1"     "niac1"     "thry1"
# [41] "istrd1"    "ostrd1"    "beta1"     "betad1"    "ccb1"      "ace1"      "aced1"     "vaso1"
# [49] "vasod1"    "diuret1"   "dig1"      "ntg1"      "diffa10"   "phctdn25"  "phacls25"  "limit25"
# [57] "exefrt25"  "emctdn25"  "emacls25"  "carful25"  "tfawda02"  "tfawea02"  "twuwda02"  "twuwea02"
# [65] "surgtr02"  "o2thpy02"  "ethnicity"

# $`3`
#  [1] "race"        "slpill15"    "nitro15"     "stress15"    "meds10"      "cigars10"
#  [7] "vigact25"    "modact25"    "lift25"      "climbs25"    "climb125"    "bend25"
# [13] "wk1ml25"     "wksblk25"    "wk1blk25"    "bathe25"     "hvsnrd02"    "stpbrt02"
# [19] "mdsa02"      "membhh02"    "smokstat_s1" "lang15"

# $`4`
#  [1] "wrhead10" "wrface10" "plstc10"  "vest10"   "issnor02" "sitrd02"  "watv02"   "sitpub02"
#  [9] "pgrcar02" "lydwn02"  "sittlk02" "sitlch02" "incar02"  "attabl02" "drive02"  "re_s1"
# [17] "educat"

# $`5`
#  [1] "mstat"    "ltdp10"   "shlg10"   "rest10"   "hwwell10" "tea10"    "soda10"   "pipe10"
#  [9] "genhth25" "cmp1yr25" "probsa25" "painin25" "hlthlm25" "sickez25" "hlthy25"  "worse25"
# [17] "exclnt25" "tfa02"    "wudnrs02" "wu2em02"  "funres02" "sleepy02" "tkpill02" "nges02"
# [25] "loudsn02" "cough02"  "cp02"     "sob02"    "sweats02" "noise02"  "painjt02" "hb02"
# [33] "legcrp02" "needbr02" "rawre_s1"

# #$`6`
#  [1] "napshr15"        "bdpain25"        "pep25"           "nrvous25"        "down25"
#  [6] "calm25"          "energ25"         "blue25"          "worn25"          "happy25"
# [11] "tired25"         "hosnr02"         "age_category_s1"


def generate_data():
    np.random.seed(31415)
    data_frame = pandas.read_csv(load_data(file='final_sleep_data.csv'), index_col=0)
    time_frame = pandas.read_csv(load_data(file='final_status.csv'), index_col=0)
    event_frame = pandas.read_csv(load_data(file='final_any_cvd.csv'), index_col=0)
    print("head of data:{}, data shape:{}".format(data_frame.head(), data_frame.shape))
    print("head of time:{}, time shape:{}".format(time_frame.head(), time_frame.shape))
    print("head of event:{}, event shape:{}".format(event_frame.head(), event_frame.shape))
    # one_hot_encoder_list = ['educat', 'gender', 'ethnicity', 'mstat', 'age_category_s1', 'race', 'smokstat_s1']
    # data_frame = one_hot_encoder(data_frame, encode=one_hot_encoder_list)

    # Preprocess
    # to_drop = ["nsrrid", "pptid", "any_cvd", "status", "folder"]
    to_drop = ["nsrrid", "pptid"]
    covariates = data_frame.columns.values
    print("covariates: ", covariates)
    assert (("any_cvd" in covariates) == False)
    assert (("status" in covariates) == False)
    assert (("folder" in covariates) == False)
    print("missing:{}".format(missing_proportion(data_frame.drop(labels=to_drop, axis=1))))
    t_data = time_frame
    e_data = event_frame
    sex_F_data = data_frame[['gender']] - 1
    dataset = data_frame.drop(labels=to_drop, axis=1)
    print("head of dataset data:{}, data shape:{}".format(dataset.head(), dataset.shape))
    print("data description:{}".format(dataset.describe()))
    covariates = np.array(dataset.columns.values)
    print("columns:{}".format(covariates))
    # encoded_indices = one_hot_indices(dataset, one_hot_encoder_list)
    encoded_indices = []

    x = np.array(dataset).reshape(dataset.shape)
    t = np.array(t_data).reshape(len(t_data))
    e = np.array(e_data).reshape(len(e_data))
    sex_F = np.array(sex_F_data).reshape(len(sex_F_data))
    print("unique_gender: ", np.unique(sex_F))

    print("x:{}, t:{}, e:{}, len:{}".format(x[0], t[0], e[0], len(t)))
    idx = np.arange(0, x.shape[0])
    print("x_shape:{}".format(x.shape))

    np.random.shuffle(idx)
    x = x[idx]
    t = t[idx]
    e = e[idx]
    sex_F = sex_F[idx]
    end_time = max(t)
    print("end_time:{}".format(end_time))
    print("observed percent:{}".format(sum(e) / len(e)))
    print("shuffled x:{}, t:{}, e:{}, len:{}".format(x[0], t[0], e[0], len(t)))
    print("sex_F percent:{}".format(sum(sex_F) / len(sex_F)))
    num_examples = int(0.80 * len(e))
    print("num_examples:{}".format(num_examples))
    train_idx = idx[0: num_examples]
    split = int((len(t) - num_examples) / 2)

    test_idx = idx[num_examples: num_examples + split]
    valid_idx = idx[num_examples + split: len(t)]
    print("test:{}, valid:{}, train:{}, all: {}".format(len(test_idx), len(valid_idx), num_examples,
                                                        len(test_idx) + len(valid_idx) + num_examples))
    # print("test_idx:{}, valid_idx:{},train_idx:{} ".format(test_idx, valid_idx, train_idx))
    imputation_values = get_train_median_mode(x=np.array(x[train_idx]), categorial=encoded_indices)
    print("imputation_values:{}".format(imputation_values))

    preprocessed = {
        'train': formatted_data(x=x, t=t, e=e, idx=train_idx, imputation_values=imputation_values),
        'test': formatted_data(x=x, t=t, e=e, idx=test_idx, imputation_values=imputation_values),
        'valid': formatted_data(x=x, t=t, e=e, idx=valid_idx, imputation_values=imputation_values)
    }
    return preprocessed


def load_data(file):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.abspath(os.path.join(dir_path, '', file))
    return path


if __name__ == '__main__':
    generate_data()
