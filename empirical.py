import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numba import njit

# ==========================================
# 1. Optimal Hyperparameter Setup 
# ==========================================

def I2(lam):
    """Computes the exact expectation E[(Z-lambda)_+^2] for Z ~ N(0,1)."""
    return (1 + lam**2) * norm.sf(lam) - lam * norm.pdf(lam)

def U2(lam, delta):
    """The threshold function to minimize."""
    return lam + np.sqrt(I2(lam) / delta)

def get_optimal_lambda(delta):
    """Numerically find lambda^* via convex optimization."""
    res = minimize(U2, x0=1.0, args=(delta,), bounds=[(0, None)])
    lam_star = res.x[0]
    L0_2 = I2(lam_star)
    return lam_star, L0_2

# ==========================================
# 2. Baseline: aGRAPA Testing by Betting
# ==========================================

@njit
def agrapa_martingale_fast(X, m, c=0.5):
    n = len(X)
    M = 1.0
    mu_hat = 0.5  
    var_hat = 0.25 
    sum_X = 0.0
    mean_w = 0.0
    M2_w = 0.0
    
    for i in range(n):
        denom = var_hat + (mu_hat - m)**2 + 1e-8
        lam_target = (mu_hat - m) / denom
        lam_lower = -c / (1.0 - m + 1e-8)
        lam_upper = c / (m + 1e-8)
        
        lam_i = lam_target
        if lam_i < lam_lower: lam_i = lam_lower
        elif lam_i > lam_upper: lam_i = lam_upper
        
        M *= (1.0 + lam_i * (X[i] - m))
        
        # Welford's Update (After taking the step)
        sum_X += X[i]
        mu_hat = sum_X / (i + 1)
        
        delta = X[i] - mean_w
        mean_w += delta / (i + 1)
        M2_w += delta * (X[i] - mean_w)
        
        if i > 0:
            var_hat = M2_w / i  
            
    return M

# ==========================================
# 3. Proposed Method: Direct Polynomial
# ==========================================

@njit
def direct_polynomial_path(X, m, lam_star):
    """Computes the exact (W_n)_+^2 for a single path with two-sided adaptation."""
    n = len(X)
    lam_i = lam_star / n
    W = 0.0 
    
    mean_w = 0.0
    M2_w = 0.0
    
    # Prior mean, perfectly predictable
    mu_hat = 0.5 
    
    for i in range(n):
        x_val = X[i]
        
        # 1. Variance Regularization
        var_hat = (0.25 + M2_w) / (1.0 + i)
        sigma_hat = np.sqrt(var_hat)
        
        # 2. Predictable Sign-Flipping (The Fix!)
        # If we expect the true mean is higher than m, direction is positive.
        # If we expect the true mean is lower than m, direction is negative.
        direction = 1.0 if (mu_hat - m) >= 0 else -1.0
        
        # 3. Scale and accumulate drift (Now it grows positively for ANY false m)
        gamma_i = direction / (np.sqrt(n) * sigma_hat)
        W += gamma_i * (x_val - m) - lam_i
        
        # 4. Update Welford's and the Predictable Mean for the next step
        delta = x_val - mean_w
        mean_w += delta / (i + 1)
        M2_w += delta * (x_val - mean_w)
        mu_hat = mean_w # Update predictable mean for step i+1
        
    W_plus = 0.0 if W < 0 else W
    return W_plus ** 2


@njit
def smoothed_direct_polynomial(X, m, lam_star, I2_star, B):
    """Rao-Blackwellized Direct Polynomial normalized to form an e-value."""
    poly_sum = 0.0
    for _ in range(B):
        X_perm = np.random.permutation(X)
        poly_sum += direct_polynomial_path(X_perm, m, lam_star)
        
    expected_W2 = poly_sum / B
    
    # Normalize by I2_star so expected value under H0 is <= 1
    return expected_W2 / I2_star

# ==========================================
# 4. Experimental Harness
# ==========================================

def run_direct_scaling_experiment():
    np.random.seed(42)
    delta = 0.05
    B = 100                 # Permutations
    num_sims = 20          # MC Simulations
    grid_size = 50         # Resolution for Lebesque measure
    n_values = [50, 100, 200, 400, 600, 800] 
    
    lam_star, L0_2 = get_optimal_lambda(delta)
    m_grid = np.linspace(0, 1, grid_size)
    dist_names = ["Symmetric (Beta 2,2)", "Skewed (Beta 1,5)", "Boundary (Bernoulli 0.5)"]
    
    results = {name: {"aGRAPA": [], "Direct Polynomial": []} for name in dist_names}
    print("Running simulations to test exact theoretical bounds...\n")
    
    for n in n_values:
        sim_measures = {name: {"aGRAPA": 0.0, "Direct Polynomial": 0.0} for name in dist_names}
        
        for sim in range(num_sims):
            distributions = {
                "Symmetric (Beta 2,2)": np.random.beta(2, 2, n),
                "Skewed (Beta 1,5)": np.random.beta(1, 5, n),
                "Boundary (Bernoulli 0.5)": np.random.binomial(1, 0.5, n).astype(float)
            }
            
            for dist_name, X in distributions.items():
                agrapa_acc, poly_acc = 0, 0
                
                for m in m_grid:
                    # Baseline
                    if agrapa_martingale_fast(X, m) < (1 / delta):
                        agrapa_acc += 1
                    
                    # Direct Polynomial E-value
                    M_poly = smoothed_direct_polynomial(X, m, lam_star, L0_2, B)
                    if M_poly < (1 / delta): 
                        poly_acc += 1
                        
                sim_measures[dist_name]["aGRAPA"] += (agrapa_acc / grid_size)
                sim_measures[dist_name]["Direct Polynomial"] += (poly_acc / grid_size)
                
        for dist_name in dist_names:
            scaled_agrapa = (sim_measures[dist_name]["aGRAPA"] / num_sims) * np.sqrt(n)
            scaled_poly = (sim_measures[dist_name]["Direct Polynomial"] / num_sims) * np.sqrt(n)
            
            results[dist_name]["aGRAPA"].append(scaled_agrapa)
            results[dist_name]["Direct Polynomial"].append(scaled_poly)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(r"Scaled Lebesgue Measure ($\text{Width} \times \sqrt{n}$) — Exact Polynomial vs Betting", fontsize=16)
    
    for i, dist_name in enumerate(dist_names):
        ax = axes[i]
        ax.plot(n_values, results[dist_name]["aGRAPA"], marker='o', label="aGRAPA", color="coral", linewidth=2)
        ax.plot(n_values, results[dist_name]["Direct Polynomial"], marker='s', label="Direct Polynomial", color="teal", linewidth=2)
        
        ax.set_title(dist_name)
        ax.set_xlabel("Sample Size (n)")
        ax.set_ylabel(r"Measure $\times \sqrt{n}$")
        ax.set_xticks(n_values)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
        
    plt.tight_layout()
    filename = "empirical_measure_scaling.png" 
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    print(f"\nPlot successfully saved to '{filename}'")

if __name__ == "__main__":
    run_direct_scaling_experiment()




