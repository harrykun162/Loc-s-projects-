CREATE TABLE Bank_churn.dbo.Bank_churn (
    CustomerId INT,
    Surname VARCHAR(100),
    CreditScore INT,
    Geography VARCHAR(50),
    Gender VARCHAR(10),
    Age INT,
    Tenure INT,
    Balance FLOAT,
    NumOfProducts INT,
    HasCrCard INT,
    IsActiveMember INT,
    EstimatedSalary FLOAT,
    Exited INT
);

BULK INSERT Bank_churn.dbo.Bank_churn
FROM "C:\Users\Loc-kun\Desktop\Knowledge\Projects\Bank churn analysis\Bank_churn_RFM.csv"
WITH (
    FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    TABLOCK
);