from irf.ensemble import wrf, wrf_reg
import numpy as np
def test1():
    X = np.random.normal(size=(100,10))
    y = np.random.choice([0, 1], size=(100,))
    for tt in [wrf, wrf_reg]:
        rf = wrf(
            n_jobs=10,
            n_estimators=400,
            max_depth=None,
            bootstrap=True,
        )
        rf.fit(
            X[:,:3],
            y,
            keep_record=False,
            K=1,
            feature_weight=None,
        )
    return

def test2():
    X = np.random.normal(size=(100,10))
    y = np.random.choice([0, 1], size=(100,))
    for tt in [wrf, wrf_reg]:
        rf = wrf(
            n_jobs=10,
            n_estimators=400,
            max_depth=None,
            bootstrap=True,
        )
        rf.fit(
            X[:,:3],
            y,
            keep_record=False,
            K=1,
            feature_weight=[1,1,0],
        )
        rf.fit(
            X[:,:],
            y,
            keep_record=False,
            K=1,
            feature_weight=None,
        )
        rf = wrf_reg(
            n_jobs=10,
            n_estimators=400,
            max_depth=None,
            bootstrap=True,
        )
    return
test1()
test2()

    