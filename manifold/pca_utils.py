from skimage import morphology, transform
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from numpy.linalg import lstsq
import time

def resize_image(image, down_sample_rate=4):
    image = transform.resize(image, (image.shape[0] // down_sample_rate, image.shape[1] // down_sample_rate))
    return image

def close_image(image, imager_pixel_spacing, max_mm_r=5, down_sample_rate=4):
    max_pixel_r = (max_mm_r / imager_pixel_spacing - 1) / down_sample_rate
    footprint = morphology.disk(max_pixel_r)
    closed_image = morphology.closing(image, footprint)
    return closed_image

def center_sequence(images):
    images_c = ((images - np.mean(images)) / np.std(images)).reshape(images.shape[0], -1)
    return images_c

class BasicManifold:
    def __init__(self):
        self.learning_time = -1
        self.test_time = -1

    def fit_time(self, X):
        # measure learning time
        start_time = time.time()
        self.fit(X)
        end_time = time.time()
        self.learning_time = end_time - start_time
    
    def fit(self, X):
        pass

    def transform(self, X):
        pass

    def get_signal(self):
        pass

    def get_test_time(self, X):
        # measure test time for one frame
        start_time = time.time()
        self.transform(X)
        self.get_signal()
        end_time = time.time()
        self.test_time = (end_time - start_time) / X.shape[0]

class CustomPCA(BasicManifold):
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device = device

    def fit(self, X):
        X_centered = center_sequence(X)
        self.pca = PCA(n_components=X_centered.shape[0])
        eigen_vectors = self.pca.fit_transform(X_centered)
        eigen_vectors = X_centered.T @ eigen_vectors
        eigen_vectors = eigen_vectors / np.linalg.norm(eigen_vectors, axis=0)
        self.eigen_vectors = eigen_vectors
    
    def transform(self, X):
        X_centered = center_sequence(X)
        self.X_centered = X_centered
        eigen_vectors = self.pca.transform(X_centered)
        eigen_vectors = X_centered.T @ eigen_vectors
        eigen_vectors = eigen_vectors / np.linalg.norm(eigen_vectors, axis=0)
        self.eigen_vectors = eigen_vectors

    def get_signal(self):
        signal = np.matmul(self.X_centered, self.eigen_vectors[:, 0])
        signal_norm = normalize_unit_norm(signal)

        return signal_norm

class KPCA(BasicManifold):
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device = device
    
    def fit(self, X):
        kernel_input = torch.from_numpy(X).to(self.device).reshape(X.shape[0], -1)
        sample_means = torch.mean(kernel_input)
        self.means = sample_means
        
        sample_stds = torch.std(kernel_input)
        sample_stds[sample_stds == 0] = 1e-12
        self.stds = sample_stds

        Xn = ((kernel_input - sample_means) / sample_stds).float()
        self.Xn_train = Xn

        K = torch.matmul(Xn, Xn.T)
        
        eigen_values, eigen_vectors = torch.linalg.eigh(K)
        eigen_values = eigen_values.flip(dims=[0]).float()
        eigen_vectors = eigen_vectors.flip(dims=[1]).float()
        
        eigen_values = eigen_values[:Xn.shape[0]]
        eigen_vectors = eigen_vectors[:, :Xn.shape[0]]
        
        eigen_values = torch.clamp(eigen_values, min=0)
        alphas = eigen_vectors / torch.sqrt(eigen_values + 1e-12)
        self.alphas = alphas

    def transform(self, X):
        kernel_input = torch.from_numpy(X).to(self.device).reshape(X.shape[0], -1)
        self.X_centered = ((kernel_input - self.means) / self.stds).float()
        K = torch.matmul(self.X_centered, self.X_centered.T)
        self.eigen_vectors = torch.matmul(self.alphas.T, self.X_centered).cpu().numpy().T
    
    def get_signal(self, normalize=True):
        signal = np.matmul(self.X_centered.cpu().numpy(), self.eigen_vectors[:, 0])
        if normalize:
            signal = normalize_unit_norm(signal)
        return signal
    
class MultiResKPCA(BasicManifold):
    def __init__(self, image_width, min_patch_size=16, distance_threshold=0.05, min_cluster_size=10, device='cuda:0'):
        super().__init__()
        
        self.image_width = image_width
        self.min_patch_size = min_patch_size
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.device = device

        self.signal_found = False

        max_r = int(np.floor(np.log2(int(image_width) / min_patch_size)))
        self.patch_sizes = min_patch_size * 2**np.arange(max_r + 1)
    
    def extract_patches(self, img, patch_size):
        H, W = img.shape
        p = patch_size
        # Number of patches along height and width
        n_patches_y = H // p
        n_patches_x = W // p
        
        patches = []
        for i in range(n_patches_y):
            for j in range(n_patches_x):
                patch = img[i*p:(i+1)*p, j*p:(j+1)*p]
                patches.append(patch)
        return np.array(patches)

    def fit_patchwise_kpca(self, sequence, patch_sizes):
        prcs = []
        kpcas = []
        for p in patch_sizes:
            sequence_patches = []
            for img in sequence:
                patches = self.extract_patches(img, p).reshape((-1, p**2))
                sequence_patches.append(patches)
            patch_per_sequence = np.transpose(sequence_patches, (1, 0, 2))
            
            for patch_xy in patch_per_sequence:
                kpca = KPCA()
                kpca.fit(patch_xy)
                kpca.transform(patch_xy)
                prc = kpca.get_signal(normalize=False)
                prcs.append(prc)
                kpcas.append(kpca)
        prcs = np.array(prcs)
        self.kpcas = kpcas
        return prcs
    
    def transform_patchwise_kpca(self, sequence, patch_sizes):
        kpca_id = 0
        patch_id = 0

        prcs = []
        for p in patch_sizes:
            sequence_patches = []
            for img in sequence:
                patches = self.extract_patches(img, p).reshape((-1, p**2))
                sequence_patches.append(patches)
            patch_per_sequence = np.transpose(sequence_patches, (1, 0, 2))

            for patch_xy in patch_per_sequence:
                if int(self.patch_cluster_labels[patch_id]) == int(self.respiratory_cluster_id):
                    kpca = self.kpcas[kpca_id]
                    kpca.transform(patch_xy)
                    prc = kpca.get_signal(normalize=False)
                    prcs.append(prc)
                
                kpca_id += 1
                patch_id += 1
            
        prcs = np.array(prcs)
        return prcs

    def compute_ncc_dist_matrix(self, X):
        n_samples = X.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                ncc = normalized_cross_coefficient(X[i], X[j])
                dist = 1 - ncc
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        np.fill_diagonal(dist_matrix, 0)
        return dist_matrix

    def fit_agglomerative_clustering(self, X, mult_dist=1.1):
        distance_threshold = self.distance_threshold
        dist_matrix = self.compute_ncc_dist_matrix(X)

        largest_cluster_size = 1
        cluster_labels = []

        while largest_cluster_size < self.min_cluster_size:
            clustering = AgglomerativeClustering(
                n_clusters=None,                
                metric='precomputed',               
                linkage='average',                   
                distance_threshold=distance_threshold
            )
            cluster_labels = clustering.fit_predict(dist_matrix)

            _, counts = np.unique(cluster_labels, return_counts=True)
            largest_cluster_size = np.max(counts)

            if largest_cluster_size < self.min_cluster_size: distance_threshold = distance_threshold * mult_dist
            self.clustering = clustering
        return cluster_labels

    def repr_signal_cluster(self, X, cluster_labels):
        unique_cluster_labels = np.unique(cluster_labels)

        cluster_signals = []
        cluster_sizes = []
        for cluster_label in unique_cluster_labels:
            ids = np.argwhere(cluster_label == cluster_labels).flatten()
            cluster_size = len(ids)
            signals = X[ids]

            cluster_signal = np.mean(signals, axis=0)

            cluster_signals.append(cluster_signal)
            cluster_sizes.append(cluster_size)
        return np.array(cluster_signals), np.array(cluster_sizes), unique_cluster_labels

    def filter_clusters(self, cluster_signals, cluster_sizes):
        breath_signal_ids = []
        r2s = []
        for idx in range(0, len(cluster_signals)):
            signal = cluster_signals[idx]
            cluster_size = cluster_sizes[idx]
            if cluster_size < 3:
                continue

            best_r2, detected = has_breathing_frequence(signal)
            if detected:
                breath_signal_ids.append(idx)
                r2s.append(best_r2)

        breath_signal_ids = np.array(breath_signal_ids)

        if len(breath_signal_ids) == 0:
            self.signal_found = False
            return cluster_signals[0], 0
        
        breath_cluster_signals = cluster_signals[breath_signal_ids]
        breath_cluster_sizes = cluster_sizes[breath_signal_ids]

        winners = np.argwhere(breath_cluster_sizes == np.amax(breath_cluster_sizes)).flatten()
        if len(winners) > 1:
            winning_cluster = -1
            best_r2 = -1
            for idx in winners:
                if r2s[idx] > best_r2:
                    winning_cluster = idx
                    best_r2 = r2s[idx]
        else: winning_cluster = winners[0]
        self.signal_found = True

        final_signal = breath_cluster_signals[winning_cluster]
        cluster_id = breath_signal_ids[winning_cluster]
        return final_signal, cluster_id

    def fit(self, X):
        patch_princ = self.fit_patchwise_kpca(X, self.patch_sizes)
        cluster_labels = self.fit_agglomerative_clustering(patch_princ)

        if len(cluster_labels) == 0:
            self.signal_found = False
            return np.zeros(X.shape[0])

        cluster_signals, cluster_sizes, uniq_cluster_labels = self.repr_signal_cluster(patch_princ, cluster_labels)
        final_signal, winning_cluster_id = self.filter_clusters(cluster_signals, cluster_sizes)

        self.patch_cluster_labels = cluster_labels
        self.respiratory_cluster_id = winning_cluster_id

        final_signal = normalize(final_signal)
        self.signal_found = True
        return final_signal
    
    def transform(self, X):
        if not self.signal_found:
            return np.zeros(X.shape[0])
        patch_princ = self.transform_patchwise_kpca(X, self.patch_sizes)
        final_signal = np.mean(patch_princ, axis=0)
        final_signal = normalize(final_signal)
        return final_signal


def has_breathing_frequence(signal, fps=15, f_low=0.1, f_high=1.0, n_freqs=100, r2_threshold=0.2):
    freqs = np.linspace(f_low, f_high, n_freqs)
    t = np.arange(len(signal)) / fps
    ss_tot = np.sum((signal - signal.mean())**2)

    r2s = np.zeros_like(freqs)
    amps = np.zeros_like(freqs)
    for i, f in enumerate(freqs):
        sin_t = np.sin(2*np.pi*f*t)
        cos_t = np.cos(2*np.pi*f*t)
        A = np.column_stack((sin_t, cos_t, np.ones_like(t)))
        coeffs, *_ = lstsq(A, signal, rcond=None)
        a, b, c = coeffs
        fit = A.dot(coeffs)
        res = signal - fit
        ss_res = np.sum(res**2)
        r2 = 1.0 - (ss_res / ss_tot)             
        amp = np.sqrt(a*a + b*b)                 
        r2s[i] = r2
        amps[i] = amp
    best_idx = np.argmax(r2s)
    best_f = freqs[best_idx]
    best_r2 = float(r2s[best_idx])
    best_amp = float(amps[best_idx])
    detected = bool(best_r2 >= r2_threshold)
    return best_r2, detected

def normalized_cross_coefficient(a, b):
    a = (a - np.mean(a)) / (np.std(a))
    b = (b - np.mean(b)) / (np.std(b))
    coef = np.correlate(a, b, 'full')
    coef /= max(len(a), len(b))

    return np.max(coef)

def normalize_unit_norm(x):
    x_centered = x - np.mean(x)
    norm = np.linalg.norm(x_centered)
    return x_centered / (norm + 1e-12)

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

