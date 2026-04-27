-- Create descriptive stats
SELECT 
    COUNT(*) AS TotalRows,
    COUNT(DISTINCT CustomerId) AS UniqueCustomers,
    MIN(CreditScore) AS MinCreditScore,
    MAX(CreditScore) AS MaxCreditScore,
    AVG(CreditScore) AS AvgCreditScore,
    MIN(Age) AS MinAge,
    MAX(Age) AS MaxAge,
    AVG(Age) AS AvgAge,
    MIN(Tenure) AS MinTenure,
    MAX(Tenure) AS MaxTenure,
    AVG(Tenure) AS AvgTenure,
    MIN(Balance) AS MinBalance,
    MAX(Balance) AS MaxBalance,
    AVG(Balance) AS AvgBalance,
    MIN(NumOfProducts) AS MinNumOfProducts,
    MAX(NumOfProducts) AS MaxNumOfProducts,
    AVG(NumOfProducts) AS AvgNumOfProducts,
    MIN(EstimatedSalary) AS MinEstimatedSalary,
    MAX(EstimatedSalary) AS MaxEstimatedSalary,
    AVG(EstimatedSalary) AS AvgEstimatedSalary
FROM Bank_churn.dbo.Bank_churn;
GO
-- 1. RFM RAW VALUE CALCULATION
-- Assign RFM scores based on the following criteria:
CREATE OR ALTER VIEW vw_rfm_raw AS
SELECT *, 
    -- R proxy : Active status with tenure
    ROUND(
        (IsActiveMember * 1.0)            -- 0 or 1
        + (Tenure / 10.0),               -- 0.0 to 1.0
    4) AS R_raw,
 
    -- F proxy : NumOfProducts with active status and having a credit card
    ROUND(
    (NumOfProducts * 0.6)
    + (IsActiveMember * 0.25)
    + (HasCrCard * 0.15),
2
    ) AS F_raw,
 
    -- M proxy : Balance weighted 70%, salary weighted 30%
    ROUND(
        (Balance * 0.70)
        + (EstimatedSalary * 0.30),
    2) AS M_raw

FROM Bank_churn.dbo.Bank_churn;
GO
-- 2. RFM QUINTILE SCORING
-- QUINTILE SCORING (1 = lowest, 5 = highest)
-- Each R, F, M raw value is ranked into quintiles using NTILE(5) window function.
-- RECENCY:  higher raw value = more engaged, so score 5 = most engaged (normal order).
-- FREQUENCY: higher = better, normal order.
-- MONETARY:  higher = better, normal order.

CREATE OR ALTER VIEW vw_rfm_scored AS
SELECT
    *,
    -- R score: 1 (least engaged) - 5 (most engaged)
    NTILE(5) OVER (ORDER BY R_raw ASC)  AS R_score,
 
    -- F score: 1 (fewest products) - 5 (most products)
    NTILE(5) OVER (ORDER BY F_raw ASC)  AS F_score,
 
    -- M score: 1 (lowest value) - 5 (highest value)
    NTILE(5) OVER (ORDER BY M_raw ASC)  AS M_score
 
FROM vw_rfm_raw;
GO

SELECT*FROM vw_rfm_scored;
GO

-- 3. COMPOSITE RFM SCORING
-- Composite = R_score + F_score + M_score (range: 3–15)
-- Tier buckets:
--   13–15: Champion
--   10–12: Loyal customer
--    7–9: Potential loyalist
--    4–6: At risk
--    3: Lost / hibernating
 
CREATE OR ALTER VIEW vw_rfm_composite AS
SELECT
    *,
    -- Composite score
    (R_score + F_score + M_score) AS RFM_Score,
 
    -- String concatenation for pattern analysis (e.g. "5-4-3")
    CONCAT(R_score, '-', F_score, '-', M_score) AS RFM_Cell,
 
    -- Segment label
    CASE
        WHEN (R_score + F_score + M_score) >= 13 THEN 'Champion'
        WHEN (R_score + F_score + M_score) >= 10 THEN 'Loyal customer'
        WHEN (R_score + F_score + M_score) >= 7  THEN 'Potential loyalist'
        WHEN (R_score + F_score + M_score) >= 4  THEN 'At risk'
        ELSE 'Lost / hibernating'
    END AS RFM_Segment,
 
    -- Priority tier for retention targeting (1 = most urgent)
    CASE
        WHEN (R_score + F_score + M_score) >= 13 THEN 5 -- protect
        WHEN (R_score + F_score + M_score) >= 10 THEN 4 -- nurture
        WHEN (R_score + F_score + M_score) >= 7  THEN 3 -- develop
        WHEN (R_score + F_score + M_score) >= 4  THEN 2 -- re-engage
        ELSE 1 -- winback / deprioritise
    END AS Retention_Priority
 
FROM vw_rfm_scored;
GO
-- 4. MASTER VIEW
CREATE OR ALTER VIEW vw_rfm_master AS
SELECT
    CustomerId,
    Surname,
    Geography,
    Gender,
    Age,
    Tenure,
    CreditScore,
    Balance,
    NumOfProducts,
    HasCrCard,
    IsActiveMember,
    EstimatedSalary,
    Exited,
 
    -- Raw proxies
    R_raw,
    F_raw,
    M_raw,
 
    -- Individual scores
    R_score,
    F_score,
    M_score,
 
    -- Composite
    RFM_Score,
    RFM_Cell,
    RFM_Segment,
    Retention_Priority
 
FROM vw_rfm_composite;
GO
-- Convert the master table
SELECT *
INTO Bank_churn.dbo.rfm_master_table
FROM vw_rfm_master;