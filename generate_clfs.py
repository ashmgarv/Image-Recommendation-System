from optimize_fit import get_ideal_clf

#generate models for LDA
for k in [10,20,30]:
    for model in ['moment', 'lbp', 'hog', 'sift']:
        for label in ['dorsal', 'palmar', 'left', 'right', 'male', 'female', 'with_acs', 'without_acs']:
            print('running for {}_{}_{}_{}.joblib'.format(model, 'lda', k, label))
            get_ideal_clf(model, label, k, 'lda')
            print('Done.')