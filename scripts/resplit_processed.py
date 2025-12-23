# 4) stratified subject split (robust with fallbacks)
subj_list = np.array(subjects)
stratify_keys = np.array([fall_bins[s] for s in subj_list], dtype=str)

from sklearn.model_selection import train_test_split

# Helper to try stratified split with fallbacks
def try_stratified_split(subjects_arr, strat_keys, test_size, random_state=42):
    """Attempt stratified split; if it fails because some classes have <2 members,
    retry with simplified keys and finally fall back to a plain random split."""
    try:
        return train_test_split(subjects_arr, test_size=test_size, random_state=random_state, stratify=strat_keys)
    except Exception as e:
        print("WARNING: stratified split on detailed keys failed:", e)
        # try simpler stratification using only the fall_frac bucket (the part before '-')
        simple_keys = np.array([k.split('-')[0] for k in strat_keys], dtype=str)
        try:
            return train_test_split(subjects_arr, test_size=test_size, random_state=random_state, stratify=simple_keys)
        except Exception as e2:
            print("WARNING: stratified split on simple keys failed:", e2)
            # final fallback: non-stratified random split
            print("Falling back to non-stratified random split.")
            return train_test_split(subjects_arr, test_size=test_size, random_state=random_state, stratify=None)

# first split: train vs temp
train_subj, temp_subj = try_stratified_split(subj_list, stratify_keys, test_size=0.30, random_state=42)

# For the temp subset compute new stratify keys for the second split
temp_keys = np.array([fall_bins[s] for s in temp_subj], dtype=str)

# second split: val vs test (attempt stratify on temp_keys with same fallback)
val_subj, test_subj = try_stratified_split(temp_subj, temp_keys, test_size=0.5, random_state=42)