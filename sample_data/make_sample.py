"""Generate the demo workbook. Run: python sample_data/make_sample.py"""
import pandas as pd

ROWS = [
    ("Discovery & requirements", "Core", 8, 12, 20),
    ("Solution architecture", "Core", 5, 8, 15),
    ("Data model", "Core", 4, 6, 11),
    ("Backend: core services", "Core", 18, 25, 45),
    ("Backend: integrations", "Core", 10, 16, 34),
    ("Frontend: main flows", "UI", 15, 22, 38),
    ("Frontend: admin", "UI", 7, 11, 20),
    ("Migration scripts", "Data", 5, 9, 22),
    ("QA cycle 1", "Core", 8, 12, 20),
    ("QA cycle 2", "Core", 5, 8, 14),
    ("UAT & fixes", "Core", 10, 15, 30),
    ("Release & handover", "Core", 3, 5, 12),
]

df = pd.DataFrame(
    ROWS, columns=["Work package", "Track", "Best case (d)", "Expected",
                   "Worst case (d)"]
)
df.to_excel("sample_data/tasks_sample.xlsx", sheet_name="Estimates", index=False)
print("wrote sample_data/tasks_sample.xlsx")
