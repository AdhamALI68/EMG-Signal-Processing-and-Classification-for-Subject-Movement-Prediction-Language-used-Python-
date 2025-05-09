import scipy.io
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def import_mat_data(file_path):
    loaded_data = scipy.io.loadmat(file_path)
    emg_data = loaded_data['emg']
    movement_stimulus = loaded_data['stimulus'].flatten()
    print(f"Muscle Data Shape: {emg_data.shape}")
    print(f"Movement Stimulus Shape: {movement_stimulus.shape}")
    return emg_data, movement_stimulus


def apply_highpass_filter(emg_data, cutoff_frequency=10, sampling_rate=100, filter_order=4):
    nyquist_frequency = 0.5 * sampling_rate
    normalized_cutoff = cutoff_frequency / nyquist_frequency
    filter_b, filter_a = butter(filter_order, normalized_cutoff, btype='high', analog=False)
    filtered_emg = filtfilt(filter_b, filter_a, emg_data, axis=0)
    return filtered_emg


def Get_timeD_Features(ch_data):
    return ch_data.flatten()


def Get_freqD_Features(signal_data):
    FourierFt_res = np.fft.fft(signal_data, axis=0)
    FourierFt = np.abs(FourierFt_res)
    mid = len(FourierFt) // 2
    FourierFt = FourierFt[:mid]
    return FourierFt.flatten()


def Get_freqD_Features_No_Flatten(signal_data):
    FourierFt_res = np.fft.fft(signal_data, axis=0)
    FourierFt = np.abs(FourierFt_res)
    mid = len(FourierFt) // 2
    FourierFt = FourierFt[:mid]
    return FourierFt

def split_trials_by_stimulus(emg_data, stimulus_data, sampling_rate, duration_per_trial=5, excluded_stimulus=0):
    stimulus_changes = np.where(np.diff(stimulus_data) != 0)[0] + 1
    trial_start_points = np.concatenate(([0], stimulus_changes))
    trial_end_points = np.concatenate((stimulus_changes, [len(stimulus_data)]))

    trial_list = []
    trial_labels = []
    required_samples = duration_per_trial * sampling_rate

    for start_point, end_point in zip(trial_start_points, trial_end_points):
        trial_segment = emg_data[start_point:end_point, :]
        trial_label = stimulus_data[start_point]

        if trial_label == excluded_stimulus:
            continue

        if len(trial_segment) > required_samples:
            trial_segment = trial_segment[:required_samples]
            trial_list.append(trial_segment)
            trial_labels.append(trial_label)

    return trial_list, trial_labels

def leave_one_out_knn_classification_method3(data_features, data_labels, k_range=range(1, 20)):
    total_trials = len(data_features)
    Kscores = {k: [] for k in k_range}

    for k in k_range:
        predicted_labels = []
        actual_labels = []

        for j in range(total_trials):
            tmask = np.ones(total_trials, dtype=bool)
            tmask[j] = False

            xTr = np.array([data_features[i] for i in range(total_trials) if tmask[i]])
            yTr = np.array([data_labels[i] for i in range(total_trials) if tmask[i]])
            xTs = np.array(data_features[j]).reshape(1, -1)
            yTs = data_labels[j]

            H = StandardScaler()
            xTr = H.fit_transform(xTr)
            xTs = H.transform(xTs)

            NewK = min(k, len(xTr))

            knn_model = KNeighborsClassifier(n_neighbors=NewK)
            knn_model.fit(xTr, yTr)
            predicted = knn_model.predict(xTs)
            predicted_labels.extend(predicted)
            actual_labels.append(yTs)

        accuracy = accuracy_score(actual_labels, predicted_labels)
        Kscores[k].append(accuracy)
        print(f"  - K: {k}, Accuracy: {accuracy:.4f}")

    return Kscores


def leave_one_out_knn_classification(data_features, data_labels, K):
    total_trials = len(data_features)
    correct_predictions = 0

    for i in range(total_trials):
        train_indices = [j for j in range(total_trials) if j != i]
        xTr = np.array([data_features[j] for j in train_indices])
        yTr = np.array([data_labels[j] for j in train_indices])
        xTs = np.array(data_features[i]).reshape(1, -1)
        yTs = data_labels[i]

        H = StandardScaler()
        xTr = H.fit_transform(xTr)
        xTs = H.transform(xTs)

        knn_model = KNeighborsClassifier(n_neighbors=K)
        knn_model.fit(xTr, yTr)
        predicted_label = knn_model.predict(xTs)

        if predicted_label[0] == yTs:
            correct_predictions += 1

    accuracy = correct_predictions / total_trials
    return accuracy


def process_time_domain_1(raw_trials, labels):
    num_ch = raw_trials[0].shape[1]
    k_values = range(1, 20)
    accuracy = []

    for ch in range(num_ch):
        ch_features = [trial[:, ch].flatten() for trial in raw_trials]
        accuracies_for_k = [leave_one_out_knn_classification(ch_features, labels, k) for k in k_values]
        accuracy.append(accuracies_for_k)

    return accuracy


def process_frequency_domain_2(raw_trials, labels):
    num_ch = raw_trials[0].shape[1]
    k_values = range(1, 20)
    accuracy = []

    for ch in range(num_ch):
        frequency_features = [Get_freqD_Features(trial[:, ch]) for trial in raw_trials]
        accuracies_for_k = [leave_one_out_knn_classification(frequency_features, labels, k) for k in k_values]
        accuracy.append(accuracies_for_k)

    return accuracy


def V_X_FORK(Kscores):
    accuracy_matrix = np.array(Kscores)

    optimal_index = np.unravel_index(np.argmax(accuracy_matrix), accuracy_matrix.shape)

    RowI = optimal_index[0] + 1
    ColI = optimal_index[1] + 1

    return RowI, ColI, accuracy_matrix[optimal_index[0]][optimal_index[1]]



def find_optimal_k_and_accuracy(Kscores):
    k_list = list(Kscores.keys())
    avg_accuracies = [np.mean(acc_scores) for acc_scores in
                      Kscores.values()]
    optimal_k = k_list[np.argmax(avg_accuracies)]
    best_avg_accuracy = max(avg_accuracies)

    return optimal_k, best_avg_accuracy


def plot_k_vs_accuracy(Kscores, feature_description):
    k_list = list(Kscores.keys())
    avg_accuracies = [np.mean(acc_scores) for acc_scores in Kscores.values()]

    plt.figure(figsize=(10, 6))
    plt.plot(k_list, avg_accuracies, 'bo-')
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Classification Accuracy')
    plt.title(f'KNN Performance vs K ({feature_description})')
    plt.grid(True)
    plt.show()

def execute_combined_feature_method3(trials, labels):
    print("\n=== Method 3: Combined Features (Time + Frequency) ===")
    combined_features = [np.concatenate([
        Get_timeD_Features(trial), Get_freqD_Features(trial)
    ]) for trial in trials]


    Kscores = leave_one_out_knn_classification_method3(combined_features, labels)
    optimal_k, best_accuracy = find_optimal_k_and_accuracy(Kscores)

    plot_k_vs_accuracy(Kscores, "Combined Features")

    return optimal_k, best_accuracy

def execute_time_domain_feature_method3(trials, labels):
    print("\n=== Method 3: Time-Domain Features Only ===")
    time_features = [Get_timeD_Features(trial) for trial in trials]

    # Perform Leave-One-Trial-Out Cross-Validation
    Kscores = leave_one_out_knn_classification_method3(time_features, labels)
    optimal_k, best_accuracy = find_optimal_k_and_accuracy(Kscores)

    # Plot K Optimization results
    plot_k_vs_accuracy(Kscores, "Time-Domain Features")

    return optimal_k, best_accuracy

def execute_frequency_domain_feature_method3(trials, labels):
    print("\n=== Method 3: Frequency-Domain Features Only ===")
    freq_features = [Get_freqD_Features(trial) for trial in trials]

    Kscores = leave_one_out_knn_classification_method3(freq_features, labels)
    optimal_k, best_accuracy = find_optimal_k_and_accuracy(Kscores)

    plot_k_vs_accuracy(Kscores, "Frequency-Domain Features")

    return optimal_k, best_accuracy


def plot_frequency_spectrum(signal, fs, channel=0, title="Frequency Spectrum"):
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1 / fs)
    spectrum = np.abs(np.fft.rfft(signal[:, channel]))

    plt.figure(figsize=(10, 6))
    plt.plot(freq, spectrum)
    plt.title(f"{title} - Channel {channel + 1}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()


try:
    emg_data, stimulus_data = import_mat_data('subject1.mat')
    emg_filtered = apply_highpass_filter(emg_data, cutoff_frequency=10, sampling_rate=100, filter_order=4)

    for i in range(3):
        plt.figure(figsize=(10, 6))

        y_min = min(np.min(emg_data[:, i]), np.min(emg_filtered[:, i]))
        y_max = max(np.max(emg_data[:, i]), np.max(emg_filtered[:, i]))

        plt.subplot(2, 1, 1)
        plt.plot(emg_data[:, i], label=f'Original Signal - Channel {i + 1}', alpha=0.7, color='green')
        plt.title(f"Original Signal - Channel {i + 1}")
        plt.xlabel("Sample Index")
        plt.ylabel("Signal Amplitude")
        plt.legend()
        plt.ylim(y_min, y_max)

        plt.subplot(2, 1, 2)
        plt.plot(emg_filtered[:, i], label=f'Filtered Signal - Channel {i + 1}', alpha=0.7, color='red')
        plt.title(f"Filtered Signal - Channel {i + 1}")
        plt.xlabel("Sample Index")
        plt.ylabel("Signal Amplitude")
        plt.legend()
        plt.ylim(y_min, y_max)

        plt.tight_layout()
        plt.show()

    for i in range(3):
        plot_frequency_spectrum(emg_data, 100, channel=i, title="Original Signal Frequency Spectrum")
        plot_frequency_spectrum(emg_filtered, 100, channel=i, title="Filtered Signal Frequency Spectrum")

    trials, labels = split_trials_by_stimulus(emg_filtered, stimulus_data, sampling_rate=100)

    best_k_freq, best_accuracy_freq = execute_frequency_domain_feature_method3(trials, labels)
    print(f"\nBest K and Accuracy for Combined Frequency Features: K={best_k_freq}, Accuracy={best_accuracy_freq:.4f}")

    best_k_time, best_accuracy_time = execute_time_domain_feature_method3(trials, labels)
    print(f"\nBest K and Accuracy for Combined Time Features: K={best_k_time}, Accuracy={best_accuracy_time:.4f}")

    best_k_combined, best_accuracy_combined = execute_combined_feature_method3(trials, labels)
    print(
        f"\nBest K and Accuracy for Combined Time and Frequency Features: K={best_k_combined}, Accuracy={best_accuracy_combined:.4f}")

    bestChTdAccuracy = process_time_domain_1(trials, labels)
    print(V_X_FORK(bestChTdAccuracy), ("Channel, K, Accuracy"))

    best_channel_fd_accuracies = process_frequency_domain_2(trials, labels)
    print(V_X_FORK(best_channel_fd_accuracies), ("Channel, K, Accuracy"))


    for ch in range(10):
        plt.figure()
        plt.plot(range(1, 20), bestChTdAccuracy[ch], marker='o', color='green')
        plt.title(f"K vs Accuracy for Time-Domain: Channel {ch + 1}")
        plt.xlabel('K')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()


    for ch in range(10):
        plt.figure()
        plt.plot(range(1, 20), best_channel_fd_accuracies[ch], marker='o',color='green')
        plt.title(f"K vs Accuracy for Frequency-Domain: Channel {ch + 1}")
        plt.xlabel('K')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()

except Exception as e:
    print(f"Error: {e}")
