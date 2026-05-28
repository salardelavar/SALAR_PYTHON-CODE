import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import xlsxwriter

# ---------- داده‌ها ----------
data = {
    "عنوان هزینه": [
        "اسکلت و سقف ساختمان (آرماتور، بتن، قالب‌بندی، سقف تیرچه بلوک)",
        "دیوار چینی (آجر، مصالح، اجرا)",
        "کچ‌کاری (گچ، مصالح، اجرا)",
        "سرامیک کف واحد و سرویس‌ها و آشپزخانه (سرامیک، چسب، بندکشی، اجرا)",
        "سنگ‌کاری مشاعات و کف و دیوار راه پله و پارکینگ (سنگ، ملات، اجرا)",
        "اجرای سنگ نما (سنگ، ملات، اجرا، درزگیر)",
        "کابینت آشپزخانه و درب و کمد دیواری (کابینت، درب داخلی، کمد دیواری)",
        "سقف کاذب و نور مخفی (سقف کاذب، نور مخفی، اجرا)",
        "جواز و مهندسین ناظر و بیمه تأمین اجتماعی (جواز، خالنظر، بیمه)",
        "نقاشی (رنگ، بتونه، اجرا)",
        "تاسیسات برقی و مکانیکی (کابل، لوله، تجهیزات، اجرا)",
        "نصب و راه‌اندازی آسانسور (عوارض، النظور، بیمه)",
        "تجهیزات آتش‌نشانی (اعلان، حریق، جعبه، کپسول و ...)",
        "انشعابات آب، برق و گاز (هزینه انشعاب و کنتور)"
    ],
    "هزینه_هر_متر_مربع_تومان": [
        7_000_000, 2_800_000, 2_300_000, 3_800_000, 2_500_000, 3_000_000,
        2_200_000, 1_400_000, 1_100_000, 1_200_000, 2_400_000, 1_800_000,
        800_000, 700_000
    ]
}

df = pd.DataFrame(data)

total_cost_per_sqm = 35_000_000

# محاسبه درصد به صورت کسر (مقدار اصلی برای اکسل)
df["درصد_کسر"] = df["هزینه_هر_متر_مربع_تومان"] / total_cost_per_sqm

# محاسبه درصد به صورت عدد 0-100 برای نمایش در کنسول و نمودار متپلاتلیب
df["درصد_نمایش"] = df["درصد_کسر"] * 100

sum_of_costs = df["هزینه_هر_متر_مربع_تومان"].sum()

# ---------- نمایش در کنسول (با درصد صحیح) ----------
print("=" * 100)
print(f"📐 Total cost per square meter: {total_cost_per_sqm:,} Tomans")
print("=" * 100)
print("\n" + "=" * 110)
print(f"{'ردیف':<5} {'عنوان هزینه':<60} {'Cost (Tomans/sqm)':<25} {'Percentage':<15}")
print("=" * 110)
for i, row in df.iterrows():
    print(f"{i+1:<5} {row['عنوان هزینه']:<60.60} {row['هزینه_هر_متر_مربع_تومان']:>22,.0f} {row['درصد_نمایش']:>14.2f}%")
print("=" * 110)
print(f"{'Total':<66} {sum_of_costs:>22,.0f} {(sum_of_costs/total_cost_per_sqm)*100:>14.2f}%")
print("=" * 110)

if sum_of_costs == total_cost_per_sqm:
    print("\n✅ Sum of costs exactly matches 35,000,000 Tomans.")
else:
    difference = total_cost_per_sqm - sum_of_costs
    print(f"\n⚠️ Sum of costs is {sum_of_costs:,} Tomans, which differs by {difference:+,} Tomans.")

# ---------- آماده‌سازی برای نمودار ----------
df_plot = df.copy().sort_values("درصد_نمایش", ascending=True)

# ---------- ۱. ذخیره در اکسل با نمودار ستونی (استفاده از ستون درصد_کسر) ----------
excel_file = "CONSTRUCTION_PROJECT_COSTS_TWO.xlsx"
with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
    # نوشتن دیتافریم (ستون‌های مورد نظر برای کاربر)
    df.to_excel(writer, sheet_name="Cost Details", index=False)
    workbook = writer.book
    worksheet = writer.sheets["Cost Details"]
    
    # ایجاد نمودار ستونی با استفاده از ستون "درصد_کسر" (مقادیر 0 تا 1)
    chart = workbook.add_chart({'type': 'column'})
    num_rows = len(df) + 1
    # ستون A = عنوان هزینه، ستون D = درصد_کسر (چهارمین ستون)
    chart.add_series({
        'name':       'Percentage',
        'categories': ['Cost Details', 1, 0, num_rows, 0],
        'values':     ['Cost Details', 1, 3, num_rows, 3],  # ستون D (0-based index 3)
    })
    chart.set_title({'name': 'درصد هر آیتم از کل هزینه هر متر مربع'})
    chart.set_x_axis({'name': 'عنوان هزینه', 'label_rotation': 45})
    chart.set_y_axis({'name': 'درصد (%)'})
    chart.set_size({'width': 900, 'height': 500})
    worksheet.insert_chart('F5', chart)
    
    # تنظیم عرض ستون‌ها
    worksheet.set_column('A:A', 60)
    worksheet.set_column('B:B', 25)
    worksheet.set_column('C:C', 15)
    worksheet.set_column('D:D', 15)

# ---------- ۲. ذخیره در PDF (بدون تغییر، چون قبلاً درست بود) ----------
with PdfPages("CONSTRUCTION_PROJECT_COSTS_TWO.pdf") as pdf:
    # صفحه اول: نمودار میلهای افقی
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(df_plot))
    ax1.barh(y_pos, df_plot["درصد_نمایش"], color='teal')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_plot["عنوان هزینه"], fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel('درصد از کل هزینه هر متر مربع (%)', fontsize=10)
    ax1.set_title('درصد هر آیتم از کل هزینه ساخت (متر مربع)', fontsize=12)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    for i, (perc, name) in enumerate(zip(df_plot["درصد_نمایش"], df_plot["عنوان هزینه"])):
        ax1.text(perc + 0.5, i, f'{perc:.2f}%', va='center', fontsize=7)
    plt.tight_layout()
    pdf.savefig(fig1)
    plt.close(fig1)
    
    # صفحه دوم: جدول
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    ax2.axis('tight')
    ax2.axis('off')
    table_data = [["ردیف", "عنوان هزینه", "هزینه (تومان/مترمربع)", "درصد"]]
    for idx, row in df.iterrows():
        table_data.append([
            idx + 1,
            row["عنوان هزینه"],
            f"{row['هزینه_هر_متر_مربع_تومان']:,.0f}",
            f"{row['درصد_نمایش']:.2f}%"
        ])
    table = ax2.table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    ax2.set_title("جدول جزئیات هزینه هر متر مربع ساختمان", fontsize=14, pad=20)
    plt.tight_layout()
    pdf.savefig(fig2)
    plt.close(fig2)
    
    # صفحه سوم: خلاصه
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.axis('off')
    summary_text = (
        f"خلاصه هزینه هر متر مربع:\n"
        f"--------------------------------\n"
        f"کل هزینه هر متر مربع: {total_cost_per_sqm:,} تومان\n"
        f"جمع هزینه‌های ذکر شده: {sum_of_costs:,} تومان\n"
        f"تطابق: {'بله' if sum_of_costs == total_cost_per_sqm else 'خیر'}\n"
        f"تعداد آیتم‌ها: {len(df)}"
    )
    ax3.text(0.1, 0.5, summary_text, transform=ax3.transAxes, fontsize=12, verticalalignment='center')
    plt.tight_layout()
    pdf.savefig(fig3)
    plt.close(fig3)

print("\n فایل PDF اصلاح‌شده: 'CONSTRUCTION_PROJECT_COSTS_TWO_per_sqm.pdf'")
print(" فایل اکسل اصلاح‌شده: 'CONSTRUCTION_PROJECT_COSTS_TWO_per_sqm.xlsx' (نمایش درصد در نمودار صحیح است)")