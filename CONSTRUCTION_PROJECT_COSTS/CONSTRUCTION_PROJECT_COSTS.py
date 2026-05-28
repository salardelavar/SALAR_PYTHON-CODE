import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import xlsxwriter

# ---------- داده‌ها و محاسبات (مشابه قبل) ----------
data = {
    "شرح فعالیت": [
        "اجرای کامل اسکلت بتن", "کود برداری", "هزینه بیمه", "دیوار چینی",
        "تهیه و اجرای سیستم گرمایش با پکیج", "تهیه و اجرای سیستم گرمایش با کولر گازی",
        "انتخاب مهندسین (دفتر طراحی - نظارت - اجرا)", "تهیه و اجرای سقف کاذب",
        "لوله کشی آب و فاضلاب", "تهیه و نصب درب و پنجره UPVC", "گرفتن پروانه ساخت",
        "تهیه و اجرای تاسیسات برقی", "تهیه و اجرای سرامیک کف", "تهیه و اجرای گچ و خاک و گچ سفید",
        "شاسی کشی و اجرای نما", "تهیه و اجرای سنگ پله", "نرده راه پله", "لوله کشی گاز",
        "محوطه سازی حیاط", "اجرای وال پست", "درب ضد سرقت واحد", "درب ضد حریق راه پله ها",
        "نصب فریم درب و پنجره", "تهیه و اجرای کاشی سرویس ها", "داربست نما",
        "عایق بندی بام و سرویس های بهداشتی", "درب نفر رو و ماشین رو حیاط", "دست کامل شیرآلات",
        "هزینه رنگ کاری", "کابینت", "هزینه نگهداری و خدماتی", "سایر هزینه های پیش‌بینی نشده و ..."
    ],
    "هزینه_کل_تومان": [
        7_599_600_000, 115_200_000, 1_448_400_000, 1_150_000_000,
        1_200_000_000, 576_000_000, 691_200_000, 662_400_000,
        300_000_000, 396_000_000, 864_000_000, 936_000_000,
        336_000_000, 800_000_000, 252_000_000, 66_000_000,
        420_000_000, 160_000_000, 129_600_000, 110_000_000,
        96_000_000, 360_000_000, 204_000_000, 32_000_000,
        78_260_000, 150_000_000, 150_000_000, 306_000_000,
        2_250_000_000, 504_000_000, 432_000_000, 0
    ]
}

df = pd.DataFrame(data)

TOTAL_COSTS = df["هزینه_کل_تومان"].sum()
TOTAL_AREAS = 6 * 120  # 720 متر مربع
df["درصد_از_کل"] = (df["هزینه_کل_تومان"] / TOTAL_COSTS) * 100
df["هزینه_به_ازای_هر_متر_مربع_تومان"] = df["هزینه_کل_تومان"] / TOTAL_AREAS

# حذف آیتم‌های صفر برای نمودار
df_plot = df[df["هزینه_کل_تومان"] > 0].copy()
df_plot = df_plot.sort_values("درصد_از_کل", ascending=True)

# ---------- ۱. ذخیره در اکسل با نمودار داخلی (xlsxwriter) ----------
excel_file = "CONSTRUCTION_PROJECT_COSTS.xlsx"
with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
    # نوشتن دیتافریم اصلی در شیت اول
    df.to_excel(writer, sheet_name="جزئیات هزینه", index=False)
    
    # دسترسی به workbook و worksheet برای اضافه کردن نمودار
    workbook = writer.book
    worksheet = writer.sheets["جزئیات هزینه"]
    
    # ایجاد نمودار ستونی (Column Chart) برای درصدها
    chart = workbook.add_chart({'type': 'column'})
    
    # محدوده داده‌ها: ستون "درصد_از_کل" (ستون D در اکسل، چون ستون A=شرح فعالیت، B=هزینه کل، C=درصد، D=تومان بر متر مربع)
    # ردیف اول هدر است، داده‌ها از ردیف ۲ تا آخر (۳۳)
    num_rows = len(df) + 1  # +1 برای هدر
    chart.add_series({
        'name':       'درصد از کل هزینه',
        'categories': ['جزئیات هزینه', 1, 0, num_rows, 0],  # ستون A (شرح فعالیت)
        'values':     ['جزئیات هزینه', 1, 2, num_rows, 2],  # ستون C (درصد_از_کل)
    })
    
    # تنظیم عنوان و محورها
    chart.set_title({'name': 'درصد هر آیتم از کل هزینه ساخت'})
    chart.set_x_axis({'name': 'شرح فعالیت', 'label_rotation': 45})
    chart.set_y_axis({'name': 'درصد (%)'})
    chart.set_size({'width': 900, 'height': 500})
    
    # درج نمودار در شیت (مثلاً از سلول F5)
    worksheet.insert_chart('F5', chart)
    
    # تنظیم عرض ستون‌ها برای خوانایی بهتر
    worksheet.set_column('A:A', 45)
    worksheet.set_column('B:B', 25)
    worksheet.set_column('C:C', 15)
    worksheet.set_column('D:D', 25)

# ---------- ۲. ذخیره در PDF با نمودار (مانند کد قبلی) ----------
with PdfPages("CONSTRUCTION_PROJECT_COSTS.pdf") as pdf:
    # صفحه اول: نمودار میلهای افقی (با متپلاتلیب)
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    y_pos = np.arange(len(df_plot))
    ax1.barh(y_pos, df_plot["درصد_از_کل"], color='steelblue')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_plot["شرح فعالیت"], fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel('درصد از کل هزینه (%)', fontsize=10)
    ax1.set_title('درصد هر آیتم از کل هزینه ساخت', fontsize=12)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    for i, (perc, name) in enumerate(zip(df_plot["درصد_از_کل"], df_plot["شرح فعالیت"])):
        ax1.text(perc + 0.5, i, f'{perc:.2f}%', va='center', fontsize=7)
    plt.tight_layout()
    pdf.savefig(fig1)
    plt.close(fig1)
    
    # صفحه دوم: جدول کامل
    fig2, ax2 = plt.subplots(figsize=(14, 20))
    ax2.axis('tight')
    ax2.axis('off')
    table_data = [["ردیف", "شرح فعالیت", "هزینه کل (تومان)", "درصد از کل", "تومان بر متر مربع"]]
    for idx, row in df.iterrows():
        table_data.append([
            idx + 1,
            row["شرح فعالیت"],
            f"{row['هزینه_کل_تومان']:,.0f}",
            f"{row['درصد_از_کل']:.2f}%",
            f"{row['هزینه_به_ازای_هر_متر_مربع_تومان']:,.0f}"
        ])
    table = ax2.table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.2, 1.5)
    ax2.set_title("جدول جزئیات هزینه‌های ساخت (ساختمان ۶ طبقه ۱۲۰ متری)", fontsize=14, pad=20)
    plt.tight_layout()
    pdf.savefig(fig2)
    plt.close(fig2)
    
    # صفحه سوم: خلاصه محاسبات
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.axis('off')
    summary_text = (
        f"خلاصه پروژه:\n"
        f"------------------------\n"
        f"کل هزینه ساخت: {TOTAL_COSTS:,.0f} تومان\n"
        f"زیربنای کل: {TOTAL_AREAS} متر مربع (۶ طبقه × ۱۲۰ متر)\n"
        f"هزینه هر متر مربع (میانگین): {TOTAL_COSTS / TOTAL_AREAS:,.0f} تومان\n"
        f"تعداد آیتم‌های با هزینه مثبت: {len(df_plot)}"
    )
    ax3.text(0.1, 0.5, summary_text, transform=ax3.transAxes, fontsize=12, verticalalignment='center')
    plt.tight_layout()
    pdf.savefig(fig3)
    plt.close(fig3)

print("\n فایل PDF اصلاح‌شده: 'CONSTRUCTION_PROJECT_COSTS_per_sqm.pdf'")
print(f" کل هزینه ساخت: {TOTAL_COSTS:,.0f} تومان")
print(f" زیربنای کل: {TOTAL_AREAS} متر مربع")
print(f" هزینه هر متر مربع (میانگین): {TOTAL_COSTS / TOTAL_AREAS:,.0f} تومان")