# Hospital ER Dashboard

This dashboard brings together emergency and hospital encounters, patient information, staff attendance, encounter finances, surgeries, prescriptions, and pharmacy inventory. It is intended to help hospital leaders and operational teams understand demand, service activity, workforce patterns, encounter-level financial performance, and inventory movement.

The report includes both encounter-level analysis and patient-level analysis. An **encounter** is one visit or episode of care; one patient can have multiple encounters. Therefore, total encounters and distinct patients answer different questions and should not be expected to match.

### Contents

1. **Hospital Overview** for the broad activity and performance picture.
2. **Patient Insights** to examine individual patient profiles and encounter history.
3. **Staff Performance** for attendance and shift patterns.
4. **Encounter Financial Analysis** for the financial results associated with patient encounters.
5. **Inventory Analysis** for pharmacy stock positions and changes between snapshots.
6. **Key Takeaway** for a concise leadership summary.

## Data inputs and model

The project’s `Data` folder contains the source CSV files and `Data_Dictionary.csv`, which describes the fields and their meanings. The model combines these dimensions and fact tables:

**Dimensions**

- `Dim_Date` - calendar dates and date attributes.
- `Dim_Department` - hospital departments.
- `Dim_Medicine` - medicine and supply descriptions, unit price, and unit cost.
- `Dim_Patient` - patient identifiers and demographic attributes.
- `Dim_Staff` - staff identifiers, roles, departments, and employment dates.
- `Dim_Ward` - ward types and bed capacity.

**Fact tables**

- `Fact_Patient_Encounters` - encounter dates, patient and staff references, admission type, diagnosis, treatment, feedback, payment status, and encounter charges/costs.
- `Fact_Staff_Attendance` - staff, department, shift type, check-in/check-out timestamps, and attendance status.
- `Fact_Surgeries` - surgery date, patient, surgeon, department, and outcome.
- `Fact_Prescriptions` - prescribed or dispensed medicine lines associated with care.
- `Fact_Pharmacy_Inventory` - medicine-level stock snapshots, receipts, sales, available quantity, unit cost, and inventory value.

## Findings

The findings below are calculated from the CSV files in `Data` and follow the report’s displayed current-period and reference-value framing. The latest encounter admission timestamp in the 2025 extract is **15 December 2025**, so 2025 is a partial year. Where the dashboard compares 2025 with 2024, the 2025 current-period totals are shown alongside 2024 full-calendar-year reference values; they are useful as dashboard context, but they are not like-for-like annual comparisons. Inventory figures use the latest available snapshot near each year end. Dollar amounts are in USD as described by the data dictionary.

### Landing Page

The landing page is the report’s entry point and directs readers to the six analysis pages: Hospital Overview, Patient Insights, Staff Performance, Encounter Financial Analysis, Inventory Analysis, and Key Takeaway.

### Hospital Overview

![Hospital Overview](<Hospital Overview.png>)
- The displayed 2025 current-period totals are **713 encounters and 533 distinct patients**. The 2024 full-year reference values are **787 encounters and 587 patients**. The current 2025 values are 74 encounters (**9.4%**) and 54 patients (**9.2%**) below those 2024 annual totals.
- Emergency encounters are **178 in 2025 versus 164 in 2024** (**+14, or 8.5%**). However, inpatient encounters are 236 versus 258 (**-22 or 8.53%** lower), and outpatient encounters are 233 versus 279 (**-46 or 16.49%** lower). The mix suggests emergency demand is holding up more strongly than inpatient and outpatient activity.
- Neurology had the most encounters in 2025 (**150; 21.0%**), while Pediatrics had the fewest (**127; 17.8%**). However, compared to last year, the number of encounter is only higher in Cardiology with 2 extra encounters (**+1.4%**)
- Male patients peak in the 50-59 age group (33), while female patients peak in the 10-19 age group (37). Counts are highest among working-age adults and youth, dropping off significantly at the oldest age brackets (90-99). 

### Patient Insights

![Patient Insights](<Patient Insight.png>)
- Mean feedback was **2.93/5** in the 2025 extract versus **3.03/5** in 2024 if all Patient IDs are selected.
- Example: Patient **P0005** (Sarah Clark in the source data) is an example of how to read the selected-patient profile. In 2025, the record shows **one emergency encounter** assigned to Orthopedics, admitted on 26 May and discharged on 2 June. The encounter has **$2,376.40 in treatment charges**, **$4,214.74 in medication charges**, **no surgery charge**, and **$6,591.14 total billed charges**. The recorded feedback score is **2.06/5**.

### Staff Performance

![Staff Performance](<Staff Performance.png>)
- Overall, the present rate of staff is 90.44%, where all staff showed up to work in **7122 shifts** out of **7875 expected shifts**. The average work duration is roughly 9.4 hours per shift. 
- The bar chart shows that number of daytime shift is equal to that of nighttime shift.
- There is also a table showing the attendance history for analysing individual staff.
- Example: Staff **NRS036**, a nurse assigned to Oncology, illustrates the individual attendance view. Across the full attendance extract, the staff member has **105 attendance records: 101 Present and 4 Absent**, giving a record-based presence rate of **96.2%**. The average shift duration of this nurse is roughly **9.4 hours**.

### Encounter Financial Analysis

![Encounter Financial Analysis](<Encounter Financial Analysis.png>)
- The analysis is conducted based on solely the charges made from each encounter, not accounting for the inventory cost, labour cost or rental matters. 
- Billed encounter charges total **$3.60 million** in 2025, compared with **$3.87 million** for full-year 2024 (**roughly $270,000, or 7.5%, lower**). The revenue sums treatment, surgery, and medication charges in `Fact_Patient_Encounters`.
- The total operation cost is **$2.52 million** in 2025 compared with **$2.71 million** in 2024. The cost is roughly **$190,000** lower in the later year.
- The source-based encounter gross margin is about **29.90%** for 2025 versus **30.16%** in 2024, a decrease of about **0.26 percentage points**.

### Inventory Analysis

![Inventory Analysis](<Inventory Analysis.png>)
- The 2025 extract contains **24 snapshots**, from 10 January to 21 December. At the first snapshot, available stock was **11,182 units** valued at **$572,159**. At the latest snapshot, it was **10,704 units** valued at **$556,356**, which is a reduction of **478 units (4.3%)** and **$15,803 (2.8%)** across those snapshots.
- Compared with the latest 2024 snapshot (26 December: **11,422 units; $590,352**), the latest 2025 snapshot (21 December) is **718 units (6.3%)** and **$33,996 (5.8%)** lower in aggregate.
- All 43 medicines had more than 100 units at the latest snapshot. The medicine with the lowest quantity of stock is Antibiotics Type 2, in which there are **108 units**. However, this does not mean the lower-stock items need replenishment, because their demand is not extreme and there is no expiry information that I can use to assess shortage and waste risk.

### Key Takeaway

The report’s 2025 values point to four management priorities:

1. **Investigate the emergency-demand shift.** Emergency encounters reached 187 versus 171 in 2024, while total 2025 encounters remain below the 2024 full-year reference. Review acuity, arrival patterns, repeat visits, and throughput before changing capacity.
2. **Improve the patient experience.** Average feedback was 2.93 compared with 3.03 in 2024. Examine comments and service-specific patterns, then track whether targeted improvements change patient experience and care quality.
3. **Protect financial sustainability.** Billed charges are lower than 2024 full-year charges, while the average billed amount per encounter is higher and the reported margin is slightly lower. Review the full service-line cost picture, including pharmacy inventory costs, before drawing profitability conclusions.
4. **Manage medicines by demand and expiry risk.** Aggregate inventory value was lower at the latest 2025 snapshot, and no item was below 100 units. Use medicine-specific consumption, replenishment times, and expiry dates to set stock controls rather than relying on total units alone.

## Interpretation notes

- **Encounters are not unique patients.** A patient may contribute more than one encounter.
- **Check the period and filters.** The latest encounter admission in the source extract reviewed for this README is 15 December 2025. The dashboard’s 2025 values are therefore partial-period figures; compare them with 2024 reference values with that coverage difference in mind.
- **Inventory is snapshot-based.** A snapshot balance is not a monthly flow. Therefore, I used stock sold/received for movement analysis.
- **A dashboard metric is not a causal explanation.** Validate definitions and operational context before taking action.
- **Patient-identifiable information is sensitive.** Share screenshots or extracts containing patient names or IDs only with approved audiences.
