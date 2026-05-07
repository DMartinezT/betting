import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm
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
    """Numba-compiled aGRAPA with regularized predictable estimators."""
    n = len(X)
    M = 1.0
    
    sum_X = 0.0
    mean_w = 0.0
    M2_w = 0.0
    
    for i in range(n):
        # 1. Regularized Predictable Estimates (The Fix!)
        # We add 1 pseudo-observation of mean=0.5 and variance=0.25
        mu_hat = (0.5 + sum_X) / (1.0 + i)
        var_hat = (0.25 + M2_w) / (1.0 + i)
        
        # 2. Calculate the aGRAPA bet
        denom = var_hat + (mu_hat - m)**2 + 1e-8
        lam_target = (mu_hat - m) / denom
        
        lam_lower = -c / (1.0 - m + 1e-8)
        lam_upper = c / (m + 1e-8)
        
        lam_i = lam_target
        if lam_i < lam_lower: lam_i = lam_lower
        elif lam_i > lam_upper: lam_i = lam_upper
        
        # 3. Accumulate Wealth
        M *= (1.0 + lam_i * (X[i] - m))
        
        # 4. Update purely predictable trackers for the NEXT step (i+1)
        sum_X += X[i]
        delta = X[i] - mean_w
        mean_w += delta / (i + 1)
        M2_w += delta * (X[i] - mean_w)
            
    return M



@njit
def horizon_betting_fast(X, m, delta, c=0.5):
    """Numba-compiled betting martingale using the horizon-dependent Bernstein bet."""
    n = len(X)
    M = 1.0
    
    sum_X = 0.0
    mean_w = 0.0
    M2_w = 0.0
    
    for i in range(n):
        # 1. Regularized Predictable Estimates
        mu_hat = (0.5 + sum_X) / (1.0 + i)
        var_hat = (0.25 + M2_w) / (1.0 + i)
        
        # 2. Horizon-Dependent Bet Magnitude (Your proposed formula)
        # Note: We use np.log to get the natural logarithm
        magnitude = np.sqrt(2.0 * np.log(2.0 / delta) / (n * var_hat))
        
        # 3. Apply Directionality
        direction = 1.0 if (mu_hat - m) >= 0 else -1.0
        lam_target = direction * magnitude
        
        # 4. Safe Domain Clipping (Ensures M >= 0)
        lam_lower = -c / (1.0 - m + 1e-8)
        lam_upper = c / (m + 1e-8)
        
        lam_i = lam_target
        if lam_i < lam_lower: lam_i = lam_lower
        elif lam_i > lam_upper: lam_i = lam_upper
        
        # 5. Accumulate Wealth
        M *= (1.0 + lam_i * (X[i] - m))
        
        # 6. Update Welford's for step i+1
        sum_X += X[i]
        diff = X[i] - mean_w
        mean_w += diff / (i + 1)
        M2_w += diff * (X[i] - mean_w)
            
    return M

# ==========================================
# 3. Proposed Method: Halted Sequential Polynomial
# ==========================================

@njit
def halted_sequential_path(X, m, lam_star, I2_star):
    """Computes the sequential martingale path, halting if ruin is imminent."""
    n = len(X)
    lam_i = lam_star / n
    W = 0.0 
    
    # 1. Initialize capital strictly to the variance budget (I2_star)
    M = I2_star  
    
    mean_w = 0.0
    M2_w = 0.0
    mu_hat = 0.5 
    
    for i in range(n):
        x_val = X[i]
        
        var_hat = (0.25 + M2_w) / (1.0 + i)
        sigma_hat = np.sqrt(var_hat)
        
        # Two-sided alignment
        direction = 1.0 if (mu_hat - m) >= 0 else -1.0
        gamma_i = direction / (np.sqrt(n) * sigma_hat)
        
        W_plus = 0.0 if W < 0 else W
        
        # ----------------------------------------------------
        # 2. THE NON-NEGATIVITY CHECK (Halting Condition)
        # ----------------------------------------------------
        # Find the absolute worst-case domain bound for the next step
        worst_x = 0.0 if direction == 1.0 else 1.0
        worst_drift = gamma_i * (worst_x - m)
        
        # Predict the wealth drop
        worst_M_next = M + 2.0 * worst_drift * W_plus
        
        if worst_M_next < 0:
            # We cannot guarantee non-negativity. 
            # Freeze the martingale and stop betting.
            break
            
        # ----------------------------------------------------
        # 3. ACCUMULATE MARTINGALE
        # ----------------------------------------------------
        M += 2.0 * gamma_i * (x_val - m) * W_plus
        
        # Accumulate the state variable W
        W += gamma_i * (x_val - m) - lam_i
        
        # Update trackers predictably for step i+1
        delta = x_val - mean_w
        mean_w += delta / (i + 1)
        M2_w += delta * (x_val - mean_w)
        mu_hat = mean_w
        
    return M

@njit
def smoothed_halted_sequential(X, m, lam_star, I2_star, B):
    """Rao-Blackwellized Halted Sequential Polynomial normalized to form an e-value."""
    M_sum = 0.0
    for _ in range(B):
        X_perm = np.random.permutation(X)
        M_sum += halted_sequential_path(X_perm, m, lam_star, I2_star)
        
    expected_M = M_sum / B
    
    # Normalize by I2_star so the expected value under H0 is exactly <= 1
    # This aligns the threshold perfectly with 1/delta.
    return expected_M / I2_star




@njit
def Mt2_sequential_path(X, m, lam_star, I2_star):
    """Computes the exact M_{t,2} supermartingale with elastic clipping."""
    n = len(X)
    lam_i = lam_star / n
    W = 0.0 
    
    # Initialize capital strictly to the variance budget
    M = I2_star  
    
    mean_w = 0.0
    M2_w = 0.0
    mu_hat = 0.5 
    
    for i in range(n):
        x_val = X[i]
        
        var_hat = (0.25 + M2_w) / (1.0 + i)
        sigma_hat = np.sqrt(var_hat)
        
        # Two-sided alignment
        direction = 1.0 if (mu_hat - m) >= 0 else -1.0
        gamma_target = direction / (np.sqrt(n) * sigma_hat)
        
        W_plus = 0.0 if W < 0 else W
        
        # ----------------------------------------------------
        # THE NON-NEGATIVITY CHECK (Elastic Clipping)
        # ----------------------------------------------------
        # Find the absolute worst-case domain bound for the next step
        worst_x = 0.0 if direction == 1.0 else 1.0
        
        # What is the maximum possible wealth we could lose this step?
        worst_raw_drift = worst_x - m
        worst_possible_change = 2.0 * gamma_target * worst_raw_drift * W_plus
        
        # If this step could bankrupt us, clip gamma to the maximum safe affordable limit
        if M + worst_possible_change < 0:
            # We add 1e-8 to the denominator to prevent division by zero
            max_safe_gamma = M / (2.0 * abs(worst_raw_drift) * W_plus + 1e-8)
            gamma_i = direction * max_safe_gamma
        else:
            gamma_i = gamma_target
            
        # ----------------------------------------------------
        # ACCUMULATE THE M_{t,2} MARTINGALE
        # ----------------------------------------------------
        M += 2.0 * gamma_i * (x_val - m) * W_plus
        
        # Accumulate the state variable W for the next step
        W += gamma_i * (x_val - m) - lam_i
        
        # Update predictable trackers for step i+1
        delta = x_val - mean_w
        mean_w += delta / (i + 1)
        M2_w += delta * (x_val - mean_w)
        mu_hat = mean_w
        
    return M

@njit
def smoothed_Mt2_sequential(X, m, lam_star, I2_star, B):
    """Rao-Blackwellized M_{t,2} Polynomial normalized to form an e-value."""
    M_sum = 0.0
    for _ in range(B):
        X_perm = np.random.permutation(X)
        M_sum += Mt2_sequential_path(X_perm, m, lam_star, I2_star)
        
    expected_M = M_sum / B
    return expected_M / I2_star



# ==========================================
# 4. Experimental Harness
# ==========================================
def run_direct_scaling_experiment():
    # ==========================================
    # EXPERIMENT CONFIGURATION
    # ==========================================
    INCLUDE_HALTED_POLYNOMIAL = False  
    
    np.random.seed(42)
    delta = 0.05
    B = 150                 
    num_sims = 100          
    grid_size = 100         
    n_values = [50, 100, 500, 1000, 1500] 
    
    lam_star, L0_2 = get_optimal_lambda(delta)
    m_grid = np.linspace(0, 1, grid_size)
    dist_names = ["Symmetric (Beta 2,2)", "Skewed (Beta 1,5)", "Boundary (Bernoulli 0.5)"]
    
    methods = ["Horizon Betting", "Mt2 Polynomial"]
    if INCLUDE_HALTED_POLYNOMIAL:
        methods.insert(1, "Halted Polynomial")
        
    # Store mean, lower, and upper bounds for plotting
    results = {name: {method: {'mean': [], 'lower': [], 'upper': []} 
               for method in methods} for name in dist_names}
    
    print(f"Running simulations... (Horizon Betting, Mt2 Polynomial, and optional Halted)\n")
    
    for n in n_values:
        # Dictionary to store results for EVERY simulation at this specific n
        n_sim_raw_data = {name: {method: [] for method in methods} for name in dist_names}
        
        for sim in tqdm(range(num_sims), desc=f"n={n}"):
            distributions = {
                "Symmetric (Beta 2,2)": np.random.beta(2, 2, n),
                "Skewed (Beta 1,5)": np.random.beta(1, 5, n),
                "Boundary (Bernoulli 0.5)": np.random.binomial(1, 0.5, n).astype(float)
            }
            
            for dist_name, X in distributions.items():
                acc = {method: 0 for method in methods}
                
                for m in m_grid:
                    # 1. Horizon-Aware Betting
                    if horizon_betting_fast(X, m=m, delta=delta) < (1 / delta):
                        acc["Horizon Betting"] += 1
                    
                    # 2. Halted Sequential
                    if INCLUDE_HALTED_POLYNOMIAL:
                        if smoothed_halted_sequential(X, m, lam_star, L0_2, B) < (1 / delta): 
                            acc["Halted Polynomial"] += 1
                        
                    # 3. Mt2 Sequential
                    if smoothed_Mt2_sequential(X, m, lam_star, L0_2, B) < (1 / delta): 
                        acc["Mt2 Polynomial"] += 1
                        
                # Store the scaled measure (Measure * sqrt(n)) for this specific simulation
                for method in methods:
                    measure = acc[method] / grid_size
                    n_sim_raw_data[dist_name][method].append(measure * np.sqrt(n))
                
        # Calculate statistics across simulations for this n
        for dist_name in dist_names:
            for method in methods:
                data = np.array(n_sim_raw_data[dist_name][method])
                results[dist_name][method]['mean'].append(np.mean(data))
                
                # Extract empirical 95% confidence bands
                low, high = np.percentile(data, [2.5, 97.5])
                results[dist_name][method]['lower'].append(low)
                results[dist_name][method]['upper'].append(high)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(r"Scaled Lebesgue Measure ($\text{Width} \times \sqrt{n}$) with 95% Empirical Bands", fontsize=16)
    
    # Define colors and markers for consistency
    style = {
        "Horizon Betting": {"color": "coral", "marker": "o", "alpha": 0.2},
        "Halted Polynomial": {"color": "teal", "marker": "s", "alpha": 0.15},
        "Mt2 Polynomial": {"color": "indigo", "marker": "^", "alpha": 0.1, "ls": "--"}
    }
    
    for i, dist_name in enumerate(dist_names):
        ax = axes[i]
        
        for method in methods:
            m_data = results[dist_name][method]
            s = style[method]
            
            # 1. Plot the Mean Line
            ax.plot(n_values, m_data['mean'], marker=s["marker"], 
                    label=method, color=s["color"], linewidth=2, 
                    linestyle=s.get("ls", "-"))
            
            # 2. Plot the Shaded Confidence Band
            ax.fill_between(n_values, m_data['lower'], m_data['upper'], 
                            color=s["color"], alpha=s["alpha"])
        
        ax.set_title(dist_name)
        ax.set_xlabel("Sample Size (n)")
        ax.set_ylabel(r"Measure $\times \sqrt{n}$")
        ax.set_xticks(n_values)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        
    plt.tight_layout()
    filename = "empirical_measure_scaling_with_bands.png" 
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved with 95% empirical bands to '{filename}'")


if __name__ == "__main__":
    run_direct_scaling_experiment()






