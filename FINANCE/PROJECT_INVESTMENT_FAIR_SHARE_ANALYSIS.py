"""
In a $1 million construction project completed over 34 months,
 five partners joined at different times and invested varying
 amounts. The goal of this model is to calculate the fair share
 of each partner in the total project investment, accounting for
 their entry time and the Time Value of Money (TVM) by incorporating
 annual interest and inflation rates. Determine each partner’s fair share, the investments are discounted
 to the project's start (Month 1) to account for the Time Value of Money
 and inflation. The sum of these discounted values is then used to calculate
 each partner’s equitable percentage contribution relative to the total
 adjusted project value. 

Partner Details:  

- Partner 1: Joined in Month 1, invested $200,000  
- Partner 2: Joined in Month 3, invested $300,000  
- Partner 3: Joined in Month 6, invested $80,000  
- Partner 4: Joined in Month 10, invested $120,000  
- Partner 5: Joined in Month 15, invested $300,000  

Economic Conditions:  

- Year 1: Interest rate 5%, Inflation rate 8%  
- Year 2: Interest rate 4.5%, Inflation rate 6%  
- Year 3: Interest rate 5%, Inflation rate 7%  

Model Output:  

- A partner details table including entry month, invested amount, present
 (discounted) value, and fair share percentage.  
- A pie chart visualizing each partner’s proportional share of the total
 adjusted investment.  


-------------------------------------------------------------
THIS PROGRAM WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI) 
EMAIL: salar.d.ghashghaei@gmail.com 
-------------------------------------------------------------
"""
#%%------------------------------------------------------------------------------
import matplotlib.pyplot as plt

def calculate_fair_shares():
    # Project details
    project_value = 1_000_000  # USD
    duration_months = 34
    
    # Partner data
    partners = [
        {"name": "Partner 1", "join_month": 1, "investment": 200_000},
        {"name": "Partner 2", "join_month": 3, "investment": 300_000},
        {"name": "Partner 3", "join_month": 6, "investment": 80_000},
        {"name": "Partner 4", "join_month": 10, "investment": 120_000},
        {"name": "Partner 5", "join_month": 15, "investment": 300_000}
    ]
    
    # Economic conditions
    economic_conditions = [
        {"year": 1, "interest_rate": 0.05, "inflation_rate": 0.08},
        {"year": 2, "interest_rate": 0.045, "inflation_rate": 0.06},
        {"year": 3, "interest_rate": 0.05, "inflation_rate": 0.07}
    ]
    
    # Calculate present values (simplified discounting for demonstration)
    total_pv = 0
    for partner in partners:
        # Simplified discount factor (in reality would use proper time-value calculations)
        months_in_project = duration_months - partner['join_month'] + 1
        discount_factor = 1 / (1 + 0.06)**(months_in_project/12)  # Using avg 6% discount rate
        partner['present_value'] = partner['investment'] * discount_factor
        total_pv += partner['present_value']
    
    # Calculate fair shares
    for partner in partners:
        partner['fair_share'] = (partner['present_value'] / total_pv) * 100
    
    return partners, total_pv
#%%------------------------------------------------------------------------------
def generate_report(partners, total_pv):
    report = f"""PROJECT INVESTMENT FAIR SHARE ANALYSIS{'='*50} PARTNER CONTRIBUTIONS AND FAIR SHARES: {'='*50} {"Name":<15} {"Join Month":>12} {"Investment":>15} {"Present Value":>15} {"Fair Share %":>15}"""
    
    for partner in partners:
        report += (f"{partner['name']:<15} {partner['join_month']:>12} "
                  f"${partner['investment']:>14,.0f} ${partner['present_value']:>14,.0f} "
                  f"{partner['fair_share']:>14.1f}%\n")
    
    report += f"\nTotal Present Value: ${total_pv:,.0f}"
    return report
#%%------------------------------------------------------------------------------
def plot_results(partners):
    plt.figure(figsize=(10, 6))
    
    # Pie chart
    plt.subplot(1, 2, 1)
    labels = [p['name'] for p in partners]
    sizes = [p['fair_share'] for p in partners]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Fair Share Distribution')
    
    # Bar chart
    plt.subplot(1, 2, 2)
    names = [p['name'] for p in partners]
    investments = [p['investment']/1000 for p in partners]  # in thousands
    present_values = [p['present_value']/1000 for p in partners]
    
    bar_width = 0.35
    index = range(len(partners))
    
    plt.bar(index, investments, bar_width, label='Original Investment')
    plt.bar([i + bar_width for i in index], present_values, bar_width, label='Present Value')
    
    plt.xlabel('Partners')
    plt.ylabel('Amount (thousands USD)')
    plt.title('Investment vs Present Value')
    plt.xticks([i + bar_width/2 for i in index], names)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('PROJECT_INVESTMENT_FAIR_SHARE_ANALYSIS.png')
    plt.show()
#%%------------------------------------------------------------------------------
def main():
    # Calculate and report
    partners, total_pv = calculate_fair_shares()
    report = generate_report(partners, total_pv)
    print(report)
    
    # Save report to file
    with open("PROJECT_INVESTMENT_FAIR_SHARE_ANALYSIS.txt", "w") as f:
        f.write(report)
    
    # Generate plots
    plot_results(partners)
    
    print("\nVisualization saved as 'PROJECT_INVESTMENT_FAIR_SHARE_ANALYSIS.png'")
#%%------------------------------------------------------------------------------
if __name__ == "__main__":
    main()