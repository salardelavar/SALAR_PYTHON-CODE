def calculate_cost_of_debt(interest_rate, tax_rate):
    return interest_rate * (1 - tax_rate)

def calculate_cost_of_equity(risk_free_rate, risk_premium):
    return risk_free_rate + risk_premium

def calculate_wacc(equity, debt, cost_of_equity, cost_of_debt):
    total = equity + debt
    wacc = (equity / total) * cost_of_equity + (debt / total) * cost_of_debt
    return wacc

# ------------------------------
# ورودی‌های کاربر:

debt_amount = 5_000_000_000         # وام
equity_amount = 5_000_000_000       # سرمایه شخصی

interest_rate = 0.30                # نرخ بهره وام
tax_rate = 0.15                     # نرخ مالیات

risk_free_rate = 0.25               # نرخ سود بانکی
risk_premium = 0.25                 # صرف ریسک ساخت‌وساز

# ------------------------------
# محاسبات:

cost_of_debt = calculate_cost_of_debt(interest_rate, tax_rate)
cost_of_equity = calculate_cost_of_equity(risk_free_rate, risk_premium)
wacc = calculate_wacc(equity_amount, debt_amount, cost_of_equity, cost_of_debt)

# ------------------------------
# خروجی:

print(f"هزینه سرمایه بدهی (بعد از مالیات): {cost_of_debt*100:.2f}%")
print(f"هزینه سرمایه سهام: {cost_of_equity*100:.2f}%")
print(f"WACC (میانگین وزنی هزینه سرمایه): {wacc*100:.2f}%")
