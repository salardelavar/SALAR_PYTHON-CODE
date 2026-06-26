import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#%%------------------------------------------

# =====================================================
# Generate sample data (choose one)
# =====================================================
# Option 1: Normal data (unbounded, symmetric)
data_normal = stats.norm(loc=5, scale=2).rvs(100000)

# Option 2: Beta data (bounded between 0 and 1)
data_beta = stats.beta(a=0.5, b=0.5, loc=0, scale=1).rvs(100000)

# Option 3: Weibull data (lifetime/failure data, positive skew)
data_weibull = stats.weibull_min(c=1.5, loc=0, scale=3).rvs(100000)

# Option 4: Lognormal data (positive, heavy right tail)
data_lognorm = stats.lognorm(s=0.8, loc=0, scale=2).rvs(100000)

# Option 5: Gamma data (positive, flexible shape)
data_gamma = stats.gamma(a=2.5, loc=0, scale=1.5).rvs(100000)

# =====================================================
# SELECT YOUR DATA HERE (uncomment one)
# =====================================================
data = data_normal + data_beta + data_weibull  # Default: Beta data to show its power
# data = data_normal
# data = data_weibull
# data = data_lognorm
# data = data_gamma
def GOONESS_OF_FIT(data):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    print(f"Data range: from {min(data):.3f} to {max(data):.3f}")
    print(f"Mean: {np.mean(data):.3f} | Std Dev: {np.std(data):.3f}")
    print(f"Skewness: {stats.skew(data):.3f} | Kurtosis: {stats.kurtosis(data):.3f}")
    
    # =====================================================
    # Extended list of distributions (18 total)
    # =====================================================
    candidate_dists = [
        # === Continuous distributions ===
        'beta',          # Bounded [0,1] - percentages, proportions
        'norm',          # Unbounded symmetric - thermal noise, measurement errors
        'expon',         # Positive, memoryless - time between events
        'weibull_min',   # Positive, flexible hazard - lifetime analysis
        'weibull_max',   # Negative skew variant
        'lognorm',       # Positive, heavy right tail - repair times, particle sizes
        'gamma',         # Positive, flexible - waiting times, rainfall
        'laplace',       # Symmetric, heavy tails - wavelet coefficients, finance
        'cauchy',        # Very heavy tails - resonant systems, impulsive noise
        'uniform',       # Bounded - quantization error, random number generation
        'logistic',      # Symmetric, heavier than normal - growth models
        't',             # Student's t - small samples, robust statistics
        'rayleigh',      # Positive - signal fading, wind speeds
        'rice',          # Positive - Rician fading (LOS + multipath)
        'fisk',          # Log-logistic - survival analysis, income distribution
        'gumbel_r',      # Extreme value (right tail) - floods, maxima
        'gumbel_l',      # Extreme value (left tail) - minima, strength
        'pareto',        # Power law - city sizes, internet traffic, wealth
    ]
    
    results = []
    print(f"\n🔍 Running Goodness-of-Fit on {len(candidate_dists)} distributions...\n")
    
    # ------------------------
    # Fit and evaluate each distribution
    # ------------------------
    for name in candidate_dists:
        try:
            dist = getattr(stats, name)
            params = dist.fit(data)
            
            # K-S test
            ks_stat, p_value = stats.kstest(data, name, args=params)
            
            # Log-likelihood and AIC
            log_likelihood = dist.logpdf(data, *params).sum()
            n_params = len(params)
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + n_params * np.log(len(data))
            
            results.append({
                'name': name,
                'params': params,
                'n_params': n_params,
                'ks_stat': ks_stat,
                'p_value': p_value,
                'aic': aic,
                'bic': bic,
                'log_likelihood': log_likelihood
            })
        except Exception as e:
            print(f"⚠️  Could not fit {name}: {str(e)[:50]}...")
            continue
    
    # =====================================================
    # Sort results by P-Value (for initial screening)
    # =====================================================
    results_pvalue = sorted(results, key=lambda x: x['p_value'], reverse=True)
    
    print("\n" + "="*90)
    print("📊 RESULTS SORTED BY P-VALUE (Higher = Better Fit)")
    print("="*90)
    print(f"{'Rank':<5} {'Distribution':<15} {'P-Value':<12} {'K-S Stat':<12} {'AIC':<12} {'Params':<30}")
    print("-"*90)
    
    for idx, res in enumerate(results_pvalue, 1):
        params_str = ", ".join([f"{p:.3f}" for p in res['params']])
        # Truncate long parameter strings
        if len(params_str) > 28:
            params_str = params_str[:25] + "..."
        print(f"{idx:<5} {res['name']:<15} {res['p_value']:.5f}    {res['ks_stat']:.5f}    {res['aic']:.2f}    ({params_str})")
    
    print("="*90)
    
    # =====================================================
    # Sort by AIC (for rigorous model selection)
    # =====================================================
    results_aic = sorted(results, key=lambda x: x['aic'])
    
    print("\n" + "="*90)
    print("📊 RESULTS SORTED BY AIC (Lower = Better, Penalizes Complex Models)")
    print("="*90)
    print(f"{'Rank':<5} {'Distribution':<15} {'AIC':<12} {'P-Value':<12} {'K-S Stat':<12} {'Params':<30}")
    print("-"*90)
    
    for idx, res in enumerate(results_aic, 1):
        params_str = ", ".join([f"{p:.3f}" for p in res['params']])
        if len(params_str) > 28:
            params_str = params_str[:25] + "..."
        print(f"{idx:<5} {res['name']:<15} {res['aic']:.2f}     {res['p_value']:.5e}    {res['ks_stat']:.5f}    ({params_str})")
    
    print("="*90)
    
    # =====================================================
    # Statistical Screening: Filter by P-Value > 0.05
    # =====================================================
    valid_results = [r for r in results if r['p_value'] > 0.05]
    
    if len(valid_results) == 0:
        print("\n⚠️  WARNING: No distribution passes the P-Value > 0.05 threshold!")
        print("   Your data may come from a mixture distribution or have outliers.")
        print("   Consider: data transformation, outlier removal, or custom distribution.")
    else:
        # Sort valid results by AIC
        valid_aic = sorted(valid_results, key=lambda x: x['aic'])
        
        print("\n" + "="*90)
        print("✅ VALID DISTRIBUTIONS (P-Value > 0.05) - Sorted by AIC")
        print("="*90)
        print(f"{'Rank':<5} {'Distribution':<15} {'AIC':<12} {'P-Value':<12} {'K-S Stat':<12} {'Params':<30}")
        print("-"*90)
        
        for idx, res in enumerate(valid_aic, 1):
            params_str = ", ".join([f"{p:.3f}" for p in res['params']])
            if len(params_str) > 28:
                params_str = params_str[:25] + "..."
            print(f"{idx:<5} {res['name']:<15} {res['aic']:.2f}     {res['p_value']:.5e}    {res['ks_stat']:.5f}    ({params_str})")
        
        print("="*90)
        
        # =====================================================
        # Best distribution selection (engineering + statistical)
        # =====================================================
        best_stat = valid_aic[0]  # Best by AIC
        best_pvalue = results_pvalue[0]  # Best by P-Value
        
        print("\n" + "="*90)
        print("🏆 FINAL RECOMMENDATION")
        print("="*90)
        print(f"📌 Best by AIC (parsimonious): **{best_stat['name']}**")
        print(f"   → AIC = {best_stat['aic']:.2f}, P-Value = {best_stat['p_value']:.4f}")
        print(f"   → Parameters: ({', '.join([f'{p:.3f}' for p in best_stat['params']])})")
        print()
        print(f"📌 Best by P-Value (pure fit): **{best_pvalue['name']}**")
        print(f"   → P-Value = {best_pvalue['p_value']:.4f}, AIC = {best_pvalue['aic']:.2f}")
        
        # Engineering-specific recommendations
        print("\n" + "💡 ENGINEERING INTERPRETATION:")
        if best_stat['name'] == 'beta':
            print("   ✅ Your data is bounded [0,1] → Good for: efficiency, purity, probability of failure")
        elif best_stat['name'] in ['weibull_min', 'weibull_max']:
            print("   ✅ Your data represents lifetime/failure → Good for: reliability engineering")
        elif best_stat['name'] in ['lognorm', 'gamma']:
            print("   ✅ Your data is positive with right skew → Good for: repair times, particle sizes")
        elif best_stat['name'] in ['norm', 'laplace']:
            print("   ✅ Your data is symmetric/unbounded → Good for: measurement noise, errors")
        elif best_stat['name'] in ['expon']:
            print("   ✅ Your data has constant hazard rate → Good for: memoryless processes")
        elif best_stat['name'] in ['rayleigh', 'rice']:
            print("   ✅ Your data represents signal/communication → Good for: fading channels")
        elif best_stat['name'] in ['gumbel_r', 'gumbel_l']:
            print("   ✅ Your data represents extremes → Good for: risk analysis, flood prediction")
        elif best_stat['name'] in ['pareto']:
            print("   ✅ Your data has power-law behavior → Good for: network traffic, heavy tails")
        elif best_stat['name'] in ['t']:
            print("   ✅ Your data has heavier tails than normal → Good for: robust statistics, small samples")
        elif best_stat['name'] in ['logistic']:
            print("   ✅ Your data follows S-curve growth → Good for: population models, forecasts")
        elif best_stat['name'] in ['fisk']:
            print("   ✅ Your data follows log-logistic → Good for: survival analysis, economics")
        else:
            print(f"   ℹ️  {best_stat['name']} - Review engineering literature for interpretation.")
    
    # =====================================================
    # Advanced Visualization
    # =====================================================
    # Select top 3 distributions for visualization
    top_3 = results_pvalue[:3]
    best_overall = results_pvalue[0]
    worst_overall = results_pvalue[-1]
    
    x_axis = np.linspace(min(data) - 0.5*max(data), max(data) + 0.5*max(data), 500)
    
    plt.figure(figsize=(16, 8))
    
    # Histogram
    plt.hist(data, bins=40, density=True, alpha=0.4, color='lime', edgecolor='black', label='Data Histogram')
    
    # Plot top 3 distributions
    colors = ['red', 'blue', 'green']
    linestyles = ['-', '--', ':']
    for i, res in enumerate(top_3):
        dist = getattr(stats, res['name'])
        try:
            pdf = dist.pdf(x_axis, *res['params'])
            # Filter out invalid values
            pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
            plt.plot(x_axis, pdf, color=colors[i], linestyle=linestyles[i], 
                    linewidth=2.5, label=f"{res['name']} (P={res['p_value']:.3f}, AIC={res['aic']:.1f})")
        except:
            continue
    
    # Plot worst distribution
    worst_dist = getattr(stats, worst_overall['name'])
    try:
        pdf = worst_dist.pdf(x_axis, *worst_overall['params'])
        pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
        plt.plot(x_axis, pdf, 'k--', linewidth=1.5, alpha=0.6, 
                label=f"Worst: {worst_overall['name']} (P={worst_overall['p_value']:.3f})")
    except:
        pass
    
    plt.xlabel('Variable Value', fontsize=13)
    plt.ylabel('Probability Density', fontsize=13)
    plt.title('Goodness-of-Fit Comparison (Top 3 Distributions)', fontsize=15, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    # =====================================================
    # Summary Statistics
    # =====================================================
    print("\n" + "="*90)
    print("📋 SUMMARY STATISTICS")
    print("="*90)
    print(f"Sample Size: {len(data):,}")
    print(f"Range: [{min(data):.4f}, {max(data):.4f}]")
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Median: {np.median(data):.4f}")
    print(f"Std Dev: {np.std(data):.4f}")
    print(f"Skewness: {stats.skew(data):.4f}  {'(Symmetric)' if abs(stats.skew(data)) < 0.5 else '(Skewed)'}")
    print(f"Kurtosis: {stats.kurtosis(data):.4f}  {'(Heavy tails)' if stats.kurtosis(data) > 1 else '(Light tails)'}")
    print("="*90)
    
    # =====================================================
    # Additional Checks
    # =====================================================
    print("\n🔍 ADDITIONAL DIAGNOSTICS:")
    print("-"*50)
    
    # Check if data is bounded
    if (min(data) >= 0) and (max(data) <= 1):
        print("✓ Data is in [0,1] → Beta distribution is physically appropriate")
    elif min(data) >= 0:
        print("✓ Data is non-negative → Weibull, Lognormal, Gamma are appropriate")
    else:
        print("✓ Data is unbounded → Normal, Laplace, Cauchy are appropriate")
    
    # Check for outliers (beyond 3 sigma)
    z_scores = np.abs(stats.zscore(data))
    outliers = np.sum(z_scores > 3)
    if outliers > 0:
        print(f"⚠️  Found {outliers} outliers (beyond 3σ) → Consider robust distributions (Cauchy, t) or outlier removal")
    else:
        print("✓ No significant outliers detected")
    
    # Check sample size
    if len(data) < 30:
        print("⚠️  Small sample size (<30) → P-Values may be unreliable; consider Student's t")
    else:
        print(f"✓ Sample size {len(data)} sufficient for reliable K-S test")

#%%------------------------------------------        
GOONESS_OF_FIT(data)        