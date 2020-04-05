
USE_ACC_CORRECTION = True
SUM_ACC = [-42.77572310263783, -170.8575616629757, -1633.801397779215]
SUM_ELT = 2751


DEFAULT_CONFIG = {
    "log_freq": 0.1,
    "velocity": {
        "dampening_activated": True,
        "dampening_frequency": 200,
        "dampening_coeffs": [0.991, 0.991, 0.97]
    },
    "complementary_filter": {
        "alpha": 0.02
    },
    "acc_correction": {
        "activated": USE_ACC_CORRECTION,
        "sum_dt": 9.622,
        "sum_elt": SUM_ELT,
        "sum_acc": SUM_ACC,
        "coeffs": [acc / float(SUM_ELT) for acc in SUM_ACC] if USE_ACC_CORRECTION else [0.0, 0.0, 0.0]
    }
}
