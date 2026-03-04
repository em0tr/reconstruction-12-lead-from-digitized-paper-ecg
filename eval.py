from ecg import ECG, TypeECG, SAMPLING_RATE, LEAD_LABELS, TOTAL_NUM_ECGS, TOTAL_NUM_LEADS


if __name__ == '__main__':
    # TODO: Make a container for the data so we don't have to load it twice
    original = ECG('output/data.npy.npz', TypeECG.ORIGINAL)
    reconstructed = ECG('output/data.npy.npz', TypeECG.RECONSTRUCTED)
    assert original.__eq__(reconstructed), "Original and reconstructed signals do not match."

    ecg_number = 0
    lead_number = 0
    TOTAL_NUM_ECGS = 2  # Don't want to go through all ECGs yet
    original.find_r_peaks(ecg_number, TOTAL_NUM_ECGS, lead_number, lead_number+1)
    original.find_ecg_peaks(ecg_number, TOTAL_NUM_ECGS, lead_number, lead_number+1, use_plotting=False, print_peaks=False)
    print(f'Original T-peaks: {original.get_t_peaks(ecg_number, lead_number)}')
    print(f'Original P-peaks: {original.get_p_peaks(ecg_number, lead_number)}')
    print(f'Original Q-peaks: {original.get_q_peaks(ecg_number, lead_number)}')
    print(f'Original S-peaks: {original.get_s_peaks(ecg_number, lead_number)}')

